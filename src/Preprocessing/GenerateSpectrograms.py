import concurrent
import torchaudio
import torchaudio.transforms as T
import os
from tqdm import tqdm
from Constants.Paths import train_audio_dir, train_spectrograms_dir, test_spectrograms_dir, test_audio_dir
from Constants.Labels import all_folders
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch


def convert_to_image(spec: torch.Tensor) -> Image.Image:
    spec_db = T.AmplitudeToDB()(spec).numpy()[0]
    img_data = (((spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())) * 255.0).astype(np.uint8) \
        if spec_db.max() - spec_db.min() != 0 else np.zeros_like(spec_db).astype(np.uint8)
    return Image.fromarray(img_data, mode='L')


def process_audio_file(audio_file: str, output_path: str, spectrogram_transform: T.MelSpectrogram, backend: str | None = None) -> bool:
    try:
        waveform, _ = torchaudio.load(str(audio_file), backend=backend)
        spec = spectrogram_transform(waveform)
        img = convert_to_image(spec)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return False


def batch_process_audio(
    audio_files: list[str], 
    output_dir: str, 
    sample_rate: int = 16000, 
    batch_size: int = 64, 
    n_fft: int = 1024, 
    n_mels: int = 128, 
    ws: int = 512, 
    hl: int = 256, 
    max_workers: int | None = None, 
    backend: str | None = None
) -> None:
    output_paths = {}

    for audio_file in audio_files:
        filename = os.path.basename(str(audio_file))
        folder = str(audio_file).split(os.path.sep)[-2]
        labelFolder = folder if folder in all_folders else ''
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_folder = os.path.join(output_dir, labelFolder)
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, output_filename)
        output_paths[str(audio_file)] = output_path
        
    spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=ws,
        hop_length=hl,
        n_mels=n_mels
    )

    for i in tqdm(range(0, len(audio_files), batch_size)):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for audio_file in audio_files[i:i+batch_size]:
                futures.append(
                    executor.submit(
                        process_audio_file, 
                        audio_file, 
                        output_paths[str(audio_file)], 
                        spectrogram_transform,
                        backend
                    )
                )
            
            for _ in concurrent.futures.as_completed(futures):
                pass


def gen_train_spectrograms(backend: str | None = None) -> None:
    """Generate spectrograms for training audio files."""
    audio_files = []
    for folder in all_folders:
        audio_files.extend([os.path.join(train_audio_dir / folder, file) for file in os.listdir(train_audio_dir / folder)])
    
    batch_process_audio(audio_files, train_spectrograms_dir, backend=backend)


def gen_test_spectrograms(backend: str | None = None) -> None:
    """Generate spectrograms for test audio files."""
    audio_files = [os.path.join(test_audio_dir, file) for file in os.listdir(test_audio_dir)]
    batch_process_audio(audio_files, test_spectrograms_dir, backend=backend)


def generate_all_spectrograms(backend: str | None = None) -> None:
    """Generate spectrograms for both training and test audio files."""
    gen_train_spectrograms(backend)
    gen_test_spectrograms(backend)
