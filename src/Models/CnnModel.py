from dataclasses import dataclass
import shutil
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import pickle
import tempfile

from Models.ModelBase import ModelBase
from torch.utils.data import DataLoader, TensorDataset
from Models.TrainingHistory import TrainingHistory
from Models.RandomUndersampler import RandomUndersampler


@dataclass
class WandbDetails:
    project: str
    experiment_name: str
    config_name: str
    artifact_name: str | None = None
    # Set init_project to False to manually call wandb.init()/wandb.finish() to be able to call train() multiple times or something
    init_project: bool = True


class CnnModel(ModelBase):
    def __init__(
        self,
        classes: list[str],
        learning_rate: float = 0.001,
        lr_decay: float = 0.0001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        classifier_dropout_1: float = 0.3,
        classifier_dropout_2: float = 0.3,
        classifier_dropout_3: float = 0.1,
        print_every: int | None = 1,
        validate_every: int = 1,
        wandb_details: WandbDetails | None = None,
        seed: int = 42
    ) -> None:
        self.classes: list[str] = classes
        self.num_classes: int = len(classes)
        self.labels_index: dict[str, int] = {v: i for i, v in enumerate(classes)}
        self.learning_rate: float = learning_rate
        self.beta_1: float = beta_1
        self.beta_2: float = beta_2
        self.eps: float = eps
        self.lr_decay: float = lr_decay
        self.classifier_dropouts: tuple[float, float, float] = \
            (classifier_dropout_1, classifier_dropout_2, classifier_dropout_3)

        self.wandb_details: WandbDetails | None = wandb_details
        self.quiet: bool = print_every is None
        self.print_every: int = print_every
        self.validate_every: int = validate_every

        self.input_shape: tuple[int, int] | None = None
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.lr_scheduler: optim.lr_scheduler.LambdaLR | None = None
        self.criterion = nn.BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.history: TrainingHistory | None = None

        self.set_random_seed(seed)

    def set_random_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        np.random.seed(seed)

    def _initialize_model(self, input_shape: tuple[int, int]) -> None:
        """Initialize stuff that depends on input shape (so that it can be inferred from data)"""
        self.input_shape = input_shape
        self.model = self.CnnModule(
            input_shape,
            self.num_classes,
            (128, 64, 32),
            self.classifier_dropouts
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            eps=self.eps
        )
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (1 - self.lr_decay) ** epoch
        )

    def _validate_x(self, X: list[np.ndarray]) -> None:
        """Validate X for shape consistency"""
        if len(X) == 0:
            raise ValueError("Empty input data provided")

        # Check that all images have shape equal to input_shape
        for i, img in enumerate(X):
            if img.shape != self.input_shape:
                raise ValueError(f"Inconsistent image shape at index {i}: expected {self.input_shape}, got {img.shape}")

    def _validate_x_and_y(self, X: list[np.ndarray], y: np.ndarray) -> None:
        """Validate X and y for shape and length consistency."""
        self._validate_x(X)

        if len(X) != len(y):
            raise ValueError(f"Number of samples in X ({len(X)}) does not match y ({y.shape})")

        # Check that y contains valid class indices
        if np.max(y) >= self.num_classes or np.min(y) < 0:
            raise ValueError(f"y contains invalid class indices. Expected range [0, {self.num_classes-1}], got range [{np.min(y)}, {np.max(y)}]")

    def _to_one_hot(self, y: np.ndarray) -> np.ndarray:
        """Convert class indices to one-hot encoded array."""
        return np.eye(self.num_classes)[y]
    
    def _from_one_hot(self, y: np.ndarray) -> np.ndarray:
        """Convert one-hot encoded array to class indices."""
        return np.argmax(y, axis=1)

    def train(
        self,
        train_data: list[tuple[np.ndarray, np.ndarray]],
        val_data: list[tuple[np.ndarray, np.ndarray]] | None = None,
        epochs: int = 10,
        batch_size: int = 32
    ) -> None:
        # Special case to have a pretty fail instead of index access error
        if len(train_data) == 0 or len(train_data[0]) == 0:
            raise ValueError("Empty training data provided")

        if self.model is None:
            # Get the first image to determine input shape
            self._initialize_model(train_data[0][0].shape)

        train_loader = self._validate_data_and_make_loader(train_data, batch_size, True)
        val_loader = self._validate_data_and_make_loader(val_data, batch_size, False) \
            if val_data is not None else None

        if self.history is None:
            self.history = TrainingHistory()

        if self.wandb_details is not None and self.wandb_details.init_project:
            wandb.init(
                project=self.wandb_details.project,
                name=self.wandb_details.experiment_name,
                config=self._get_config_for_wandb(),
                settings=wandb.Settings(silent=True)
            )

        for epoch in range(epochs):
            epoch_train_loss, epoch_train_accuracy = self._train_epoch(train_loader)
            epoch_val_loss, epoch_val_accuracy = self._perform_validation(val_loader, epoch)
            
            if self.wandb_details is not None:
                wandb.log({
                    'train_loss': epoch_train_loss,
                    'train_accuracy': epoch_train_accuracy,
                    'val_loss': epoch_val_loss,
                    'val_accuracy': epoch_val_accuracy
                })

            # Update learning rate
            self.lr_scheduler.step()

            if not self.quiet and ((epoch+1) % self.print_every == 0 or epoch == epochs-1):
                self._print_metrics(
                    epoch,
                    epochs,
                    (epoch_train_loss, epoch_train_accuracy),
                    (epoch_val_loss, epoch_val_accuracy))
                
        if self.wandb_details is not None and self.wandb_details.init_project:
            if self.wandb_details.artifact_name is not None:
                self.save_model_to_wandb(self.wandb_details.artifact_name)

            wandb.finish()

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Single epoch of training, without validation/printing/anything else"""
        self.model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets_one_hot in train_loader:
            inputs = inputs.to(self.device)
            targets_one_hot = targets_one_hot.to(self.device)
            
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets_one_hot)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.detach().item()
            _, predicted = outputs.max(1)
            total_train += targets_one_hot.size(0)
            correct_train += predicted.eq(targets_one_hot.argmax(dim=1)).sum().item()

        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_accuracy = 100.0 * correct_train / total_train

        # Save training metrics
        self.history.train_loss.append(epoch_train_loss)
        self.history.train_accuracy.append(epoch_train_accuracy)

        return epoch_train_loss, epoch_train_accuracy

    def _validate_data_and_make_loader(
        self, 
        data: tuple[list[np.ndarray], np.ndarray], 
        batch_size: int, 
        undersample: bool = True
    ) -> DataLoader:
        """Validate (X, y) pair and make DataLoader for it"""
        X, y = data
        self._validate_x_and_y(X, y)

        y_one_hot = self._to_one_hot(y)

        # Convert numpy arrays to PyTorch tensors for training data
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_one_hot, dtype=torch.float32).to(self.device)

        # Create data loaders for training data
        dataset = TensorDataset(X_tensor, y_tensor)
        undersampler = RandomUndersampler(torch.from_numpy(y).to(self.device)) if undersample else None
        return DataLoader(dataset, batch_size=batch_size, sampler=undersampler)

    def _perform_validation(self, val_loader: DataLoader | None, epoch: int) -> tuple[float, float] | None:
        """Perform validation if needed"""
        if val_loader is None or (epoch + 1) % self.validate_every != 0:
            self.history.val_loss.append(None)
            self.history.val_accuracy.append(None)
            return None

        self.model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets_one_hot in val_loader:
                inputs = inputs.to(self.device)
                targets_one_hot = targets_one_hot.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets_one_hot)

                val_loss += loss.detach().item()
                _, predicted = outputs.max(1)
                total_val += targets_one_hot.size(0)
                correct_val += predicted.eq(targets_one_hot.argmax(dim=1)).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100.0 * correct_val / total_val

        self.history.val_loss.append(epoch_val_loss)
        self.history.val_accuracy.append(epoch_val_accuracy)
        return epoch_val_loss, epoch_val_accuracy

    def _print_metrics(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: tuple[float, float],
        val_metrics: tuple[float, float] | None
    ) -> None:
        if val_metrics is None:
            print(f'Epoch {epoch+1:>{len(str(total_epochs))}}/{total_epochs} | '
                  f'Train Loss: {train_metrics[0]:.4f} | '
                  f'Train Acc: {train_metrics[1]:6.2f}%')
        else:
            print(f'Epoch {epoch+1:>{len(str(total_epochs))}}/{total_epochs} | '
                  f'Train Loss: {train_metrics[0]:.4f} | '
                  f'Train Acc: {train_metrics[1]:6.2f}% | '
                  f'Val Loss: {val_metrics[0]:.4f} | '
                  f'Val Acc: {val_metrics[1]:6.2f}%')

    def predict(self, X: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        self._validate_x(X)

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            all_predicted = []
            for inputs in loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predicted.append(predicted.cpu())

        # Return predictions as numpy array
        return torch.cat(all_predicted).numpy()

    def get_history(self) -> TrainingHistory:
        return self.history

    def get_state_dict(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        return {
            'input_shape': self.input_shape,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'classes': self.classes,
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'eps': self.eps,
            'classifier_dropouts': self.classifier_dropouts,
            'history': self.history,
            'wandb_details': self.wandb_details,
            'print_every': self.print_every,
            'validate_every': self.validate_every
        }
        
    def _get_config_for_wandb(self) -> dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'eps': self.eps,
            'classifier_dropout_1': self.classifier_dropouts[0],
            'classifier_dropout_2': self.classifier_dropouts[1],
            'classifier_dropout_3': self.classifier_dropouts[2]
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "CnnModel":
        model = CnnModel(
            classes=state_dict['classes'],
            learning_rate=state_dict['learning_rate'],
            lr_decay=state_dict['lr_decay'],
            beta_1=state_dict['beta_1'],
            beta_2=state_dict['beta_2'],
            eps=state_dict['eps'],
            classifier_dropout_1=state_dict['classifier_dropouts'][0],
            classifier_dropout_2=state_dict['classifier_dropouts'][1],
            classifier_dropout_3=state_dict['classifier_dropouts'][2],
            print_every=state_dict['print_every'],
            validate_every=state_dict['validate_every'],
            wandb_details=state_dict['wandb_details']
        )

        model._initialize_model(state_dict['input_shape'])
        model.model.load_state_dict(state_dict['model_state_dict'])
        model.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        model.lr_scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        model.history = state_dict['history']

        return model

    def save_model_to_wandb(self, name: str) -> None:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "model.pkl")
        
        with open(temp_file_path, "wb") as f:
            pickle.dump(self.get_state_dict(), f)
        
        artifact = wandb.Artifact(name=name, type="model", description="Model state dict")
        artifact.add_file(temp_file_path)
        
        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait()
        
        shutil.rmtree(temp_dir)

    class ConvBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ) -> None:
            super().__init__()

            self.block = nn.Sequential(
                # First convolutional layer
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                # Second convolutional layer
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                # Max pooling
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class FeatureExtractor(nn.Module):
        def __init__(
            self,
            input_shape: tuple[int, int],
            input_channels: int = 1,
        ) -> None:
            super().__init__()
            self.input_shape = input_shape

            # Create three convolutional blocks with increasing number of channels
            self.block1 = CnnModel.ConvBlock(input_channels, 16)
            self.block2 = CnnModel.ConvBlock(16, 32)
            self.block3 = CnnModel.ConvBlock(32, 64)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x

        def get_output_size(self) -> tuple[int, int, int]:
            # Each block halves the size of the input
            return 64, self.input_shape[0] // 8, self.input_shape[1] // 8

    class Classifier(nn.Module):
        def __init__(
            self,
            input_size: int,
            num_classes: int,
            hidden_dims: tuple[int, int, int] = (128, 64, 32),
            dropout_rates: tuple[float, float, float] = (0.3, 0.3, 0.3)
        ) -> None:
            super().__init__()

            self.classifier = nn.Sequential(
                nn.Flatten(),
                # First block
                nn.Linear(input_size, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rates[0]),

                # Second block
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rates[1]),

                # Third block
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.BatchNorm1d(hidden_dims[2]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rates[2]),

                # Output layer
                nn.Linear(hidden_dims[2], num_classes),
                nn.Softmax(dim=1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(x)

    class CnnModule(nn.Module):
        def __init__(
            self,
            input_shape: tuple[int, int],
            num_classes: int,
            classifier_hidden_dims: tuple[int, int, int] = (128, 64, 32),
            classifier_dropout_rates: tuple[float, float, float] = (0.3, 0.3, 0.3)
        ) -> None:
            super().__init__()

            self.feature_extractor = CnnModel.FeatureExtractor(input_shape, input_channels=1)

            final_channels, conv_output_height, conv_output_width = self.feature_extractor.get_output_size()
            flattened_size = final_channels * conv_output_height * conv_output_width

            self.classifier = CnnModel.Classifier(
                flattened_size,
                num_classes,
                classifier_hidden_dims,
                classifier_dropout_rates
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.feature_extractor(x)
            output = self.classifier(features)
            return output
