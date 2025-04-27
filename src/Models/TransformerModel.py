from typing import Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import pickle
import tempfile
import math
import shutil

from Models.ModelBase import ModelBase
from torch.utils.data import DataLoader, TensorDataset
from Models.TrainingHistory import TrainingHistory
from Models.RandomUndersampler import RandomUndersampler # Keep for potential future use, though less common for transformers
from Models.WandbDetails import WandbDetails


# TODO: Implement proper positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(ModelBase):
    def __init__(
        self,
        classes: list[str],
        # --- Transformer specific ---
        d_model: int = 512, # Embedding dimension
        nhead: int = 8,     # Number of attention heads
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048, # Dimension in FFN
        dropout: float = 0.1,
        # --- Optimizer specific ---
        learning_rate: float = 0.0001, # Transformers often need lower LR
        lr_decay: float = 0.0,       # LR decay might be handled differently (e.g., Noam schedule)
        beta_1: float = 0.9,
        beta_2: float = 0.98, # Common beta2 for transformers
        eps: float = 1e-9,    # Common eps for transformers
        # --- Training specific ---
        print_every: int | None = 1,
        validate_every: int = 1,
        wandb_details: WandbDetails | None = None,
        seed: int = 42
    ) -> None:
        self.classes: list[str] = classes
        self.num_classes: int = len(classes) # Output dimension
        self.labels_index: dict[str, int] = {v: i for i, v in enumerate(classes)}
        
        # Transformer HParams
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Optimizer HParams
        self.learning_rate: float = learning_rate
        self.beta_1: float = beta_1
        self.beta_2: float = beta_2
        self.eps: float = eps
        self.lr_decay: float = lr_decay # May need adjustment later

        # Training config
        self.wandb_details: WandbDetails | None = wandb_details
        self.quiet: bool = print_every is None
        self.print_every: int = print_every if print_every is not None else 1
        self.validate_every: int = validate_every

        # Model state - initialized later
        self.input_feature_dim: int | None = None # Input feature dimension (e.g., MFCC count)
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.lr_scheduler: optim.lr_scheduler.LambdaLR | None = None # Or other scheduler like Noam
        # self.criterion = nn.CrossEntropyLoss() # More suitable for sequence classification/generation
        self.criterion = nn.BCELoss() # Keeping BCELoss for now, assuming similar output structure to CNN for simplicity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.history: TrainingHistory | None = None

        self.set_random_seed(seed)

    def set_random_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        np.random.seed(seed)

    def _initialize_model(self, input_feature_dim: int, max_seq_len: int) -> None:
        """Initialize stuff that depends on input shape."""
        # Note: Transformers typically don't depend on input *sequence length* for the model definition itself,
        # but might need max_seq_len for positional encoding if not learned.
        # The crucial part is the feature dimension.
        self.input_feature_dim = input_feature_dim
        self.model = self.TransformerModule(
            input_dim=self.input_feature_dim,
            output_dim=self.num_classes, # Assuming output is still num_classes like CNN
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, # Not used in this simplified version
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_len=max_seq_len # For Positional Encoding
        ).to(self.device)

        # TODO: Consider AdamW and a more sophisticated scheduler (Noam)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            eps=self.eps
        )
        # Simple decay for now
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (1 - self.lr_decay) ** epoch if self.lr_decay > 0 else 1.0
        )

    def _pad_sequences(self, sequences: list[np.ndarray], max_len: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences to the same length and create a mask."""
        # Assumes input shape: (seq_len, features)
        # Output shape: (batch, seq_len, features)
        # Mask shape: (batch, seq_len) - True where padded
        if max_len is None:
            max_len = max(seq.shape[0] for seq in sequences)
        
        feature_dim = sequences[0].shape[1]
        padded_sequences = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)
        mask = np.ones((len(sequences), max_len), dtype=bool) # True means use, False means ignore (padding)

        for i, seq in enumerate(sequences):
            length = seq.shape[0]
            padded_sequences[i, :length, :] = seq
            mask[i, length:] = False # Mark padding positions as False

        return torch.from_numpy(padded_sequences).to(self.device), torch.from_numpy(mask).to(self.device)

    def _validate_x(self, X: list[np.ndarray]) -> None:
        """Validate X for shape consistency (feature dimension)."""
        if len(X) == 0:
            raise ValueError("Empty input data provided")

        feature_dim = X[0].shape[1]
        for i, seq in enumerate(X):
            if len(seq.shape) != 2:
                 raise ValueError(f"Expected 2D array (seq_len, features) at index {i}, got {seq.shape}")
            if seq.shape[1] != feature_dim:
                raise ValueError(f"Inconsistent feature dimension at index {i}: expected {feature_dim}, got {seq.shape[1]}")
        
        # Initialize if needed, using the determined feature dimension
        if self.model is None:
             # Determine max_len from the input data for Positional Encoding
             # This might not be ideal, usually PE max_len is predefined
            max_len = max(seq.shape[0] for seq in X) 
            self._initialize_model(feature_dim, max_len)
        elif self.input_feature_dim != feature_dim:
             raise ValueError(f"Input feature dimension mismatch: model expects {self.input_feature_dim}, got {feature_dim}")


    def _validate_x_and_y(self, X: list[np.ndarray], y: np.ndarray) -> None:
        """Validate X and y for shape and length consistency."""
        self._validate_x(X) # Also handles model initialization

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
        if len(train_data) == 0 or len(train_data[0]) == 0:
            raise ValueError("Empty training data provided")

        # Unzip data
        X_train, y_train = zip(*train_data)
        X_train, y_train = list(X_train), np.array(y_train)

        # Validate and initialize model based on training data
        self._validate_x_and_y(X_train, y_train)

        # Prepare validation data if provided
        X_val, y_val = None, None
        if val_data is not None and len(val_data) > 0:
            X_val, y_val = zip(*val_data)
            X_val, y_val = list(X_val), np.array(y_val)
            self._validate_x_and_y(X_val, y_val) # Use validation data shapes only for validation

        # Create DataLoaders (manual batching for now due to padding)
        # TODO: Integrate padding into DataLoader/collate_fn for efficiency
        
        if self.history is None:
            self.history = TrainingHistory()

        if self.wandb_details is not None and self.wandb_details.init_project:
            wandb.init(
                project=self.wandb_details.project,
                name=self.wandb_details.experiment_name,
                config=self._get_config_for_wandb(),
                settings=wandb.Settings(silent=True)
            )

        num_train_samples = len(X_train)
        indices = np.arange(num_train_samples)

        for epoch in range(epochs):
            self.model.train()
            np.random.shuffle(indices) # Shuffle data each epoch
            
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            # Manual batching
            for i in range(0, num_train_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, num_train_samples)]
                batch_X = [X_train[j] for j in batch_indices]
                batch_y = y_train[batch_indices]

                # Pad batch sequences
                # TODO: Use consistent max_len if possible, maybe from self.model.max_seq_len
                max_len_batch = max(seq.shape[0] for seq in batch_X) 
                inputs, src_key_padding_mask = self._pad_sequences(batch_X, max_len=max_len_batch)
                
                # Target needs to be one-hot for BCELoss
                targets_one_hot = torch.tensor(self._to_one_hot(batch_y), dtype=torch.float32).to(self.device)

                # print(f"Input shape: {inputs.shape}") # B, S, F
                # print(f"Mask shape: {src_key_padding_mask.shape}") # B, S
                # print(f"Target shape: {targets_one_hot.shape}") # B, C

                self.optimizer.zero_grad()

                # Forward pass - adapt based on TransformerModule output
                # Assuming model outputs shape (B, C) after processing sequence
                # Need to adjust TransformerModule's forward if it outputs (S, B, C)
                outputs = self.model(inputs, src_key_padding_mask) # Pass mask
                
                # print(f"Output shape: {outputs.shape}") # B, C

                loss = self.criterion(outputs, targets_one_hot)

                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_train_loss += loss.detach().item() * len(batch_indices) # Weighted by batch size
                _, predicted = outputs.max(1)
                total_train += targets_one_hot.size(0)
                correct_train += predicted.eq(targets_one_hot.argmax(dim=1)).sum().item()

            # Calculate training metrics
            epoch_train_loss /= num_train_samples
            epoch_train_accuracy = 100.0 * correct_train / total_train
            self.history.train_loss.append(epoch_train_loss)
            self.history.train_accuracy.append(epoch_train_accuracy)

            # --- Validation ---
            epoch_val_loss, epoch_val_accuracy = None, None
            if X_val is not None and y_val is not None and (epoch + 1) % self.validate_every == 0:
                epoch_val_loss, epoch_val_accuracy = self._perform_validation(X_val, y_val, batch_size)
            
            self.history.val_loss.append(epoch_val_loss)
            self.history.val_accuracy.append(epoch_val_accuracy)

            # Log to Wandb
            if self.wandb_details is not None:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'train_accuracy': epoch_train_accuracy,
                    'learning_rate': self.optimizer.param_groups[0]['lr'] 
                }
                if epoch_val_loss is not None:
                    log_dict['val_loss'] = epoch_val_loss
                if epoch_val_accuracy is not None:
                    log_dict['val_accuracy'] = epoch_val_accuracy
                wandb.log(log_dict)


            # Update learning rate
            self.lr_scheduler.step()

            if not self.quiet and ((epoch+1) % self.print_every == 0 or epoch == epochs-1):
                self._print_metrics(
                    epoch,
                    epochs,
                    (epoch_train_loss, epoch_train_accuracy),
                    (epoch_val_loss, epoch_val_accuracy) if epoch_val_loss is not None else None)
                
        if self.wandb_details is not None and self.wandb_details.init_project:
            if self.wandb_details.artifact_name is not None:
                self.save_model_to_wandb(self.wandb_details.artifact_name)

            wandb.finish()

    # Note: _train_epoch is integrated into the main train loop for simplicity with manual batching

    # Note: _validate_data_and_make_loader replaced by manual batching in train/validation loops

    def _perform_validation(self, X_val: list[np.ndarray], y_val: np.ndarray, batch_size: int) -> tuple[float, float]:
        """Perform validation"""
        self.model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = len(y_val)

        with torch.no_grad():
            for i in range(0, total_val, batch_size):
                batch_X = X_val[i:min(i + batch_size, total_val)]
                batch_y = y_val[i:min(i + batch_size, total_val)]
                
                # Pad batch sequences
                max_len_batch = max(seq.shape[0] for seq in batch_X) 
                inputs, src_key_padding_mask = self._pad_sequences(batch_X, max_len=max_len_batch)
                targets_one_hot = torch.tensor(self._to_one_hot(batch_y), dtype=torch.float32).to(self.device)

                outputs = self.model(inputs, src_key_padding_mask)
                loss = self.criterion(outputs, targets_one_hot)

                val_loss += loss.detach().item() * len(batch_X) # Weighted average
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(targets_one_hot.argmax(dim=1)).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_accuracy = 100.0 * correct_val / total_val
        return epoch_val_loss, epoch_val_accuracy

    def _print_metrics(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: tuple[float, float],
        val_metrics: tuple[float, float] | None
    ) -> None:
        lr = self.optimizer.param_groups[0]['lr']
        epoch_str = f'Epoch {epoch+1:>{len(str(total_epochs))}}/{total_epochs}'
        train_str = f'Train Loss: {train_metrics[0]:.4f} | Train Acc: {train_metrics[1]:6.2f}%'
        val_str = ''
        if val_metrics is not None:
             val_str = f'Val Loss: {val_metrics[0]:.4f} | Val Acc: {val_metrics[1]:6.2f}%'
        lr_str = f'LR: {lr:.6f}'
        
        print(f'{epoch_str} | {train_str} | {val_str} | {lr_str}')


    def predict(self, X: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        self._validate_x(X) # Ensures model is initialized and feature dim matches

        self.model.eval()
        all_predicted = []
        total_samples = len(X)

        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_X = X[i:min(i + batch_size, total_samples)]
                
                # Pad batch sequences
                max_len_batch = max(seq.shape[0] for seq in batch_X)
                inputs, src_key_padding_mask = self._pad_sequences(batch_X, max_len=max_len_batch)
                
                outputs = self.model(inputs, src_key_padding_mask) # B, C
                _, predicted = torch.max(outputs, 1) # Get class index
                all_predicted.append(predicted.cpu())

        return torch.cat(all_predicted).numpy()

    def get_history(self) -> TrainingHistory:
        if self.history is None:
            # Return an empty history if training hasn't happened
             return TrainingHistory()
        return self.history

    def get_state_dict(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("Model has not been initialized or trained yet")

        return {
            # Model HParams
            'input_feature_dim': self.input_feature_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'max_seq_len': self.model.max_seq_len, # Get from inner model
            # Optimizer HParams
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'eps': self.eps,
            # Training Config
            'classes': self.classes,
            'history': self.history,
            'wandb_details': self.wandb_details,
            'print_every': self.print_every if not self.quiet else None,
            'validate_every': self.validate_every,
            # Model State
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        
    def _get_config_for_wandb(self) -> dict[str, Any]:
        # Consolidate HParams for logging
        config = {
            # Model
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            # Optimizer
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay, # Potentially replace with scheduler type
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'eps': self.eps,
             # Data/Input - useful context
            'input_feature_dim': self.input_feature_dim, 
            'num_classes': self.num_classes,
            'max_seq_len': self.model.max_seq_len if self.model else None, # Log if initialized
        }
        return config


    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "TransformerModel":
        # Create instance with saved HParams
        model = TransformerModel(
             # Model HParams
            classes=state_dict['classes'],
            d_model=state_dict['d_model'],
            nhead=state_dict['nhead'],
            num_encoder_layers=state_dict['num_encoder_layers'],
            num_decoder_layers=state_dict['num_decoder_layers'],
            dim_feedforward=state_dict['dim_feedforward'],
            dropout=state_dict['dropout'],
             # Optimizer HParams
            learning_rate=state_dict['learning_rate'],
            lr_decay=state_dict.get('lr_decay', 0.0), # Handle potential missing key if added later
            beta_1=state_dict['beta_1'],
            beta_2=state_dict['beta_2'],
            eps=state_dict['eps'],
             # Training Config
            print_every=state_dict.get('print_every'), # Use .get for safer loading
            validate_every=state_dict.get('validate_every', 1),
            wandb_details=state_dict.get('wandb_details') 
        )

        # Initialize the inner model structure using loaded dimensions
        model._initialize_model(
            state_dict['input_feature_dim'], 
            state_dict['max_seq_len']
        )
        
        # Load the learned weights and optimizer/scheduler states
        model.model.load_state_dict(state_dict['model_state_dict'])
        model.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        # Handle potential missing scheduler state for backward compatibility
        if 'scheduler_state_dict' in state_dict:
             model.lr_scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        # Load history if available
        model.history = state_dict.get('history')

        return model

    def save_model_to_wandb(self, name: str) -> None:
        if self.wandb_details is None or not wandb.run:
             print("Wandb not initialized or run not active. Skipping artifact saving.")
             return
             
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "transformer_model.pkl")
        
        try:
            with open(temp_file_path, "wb") as f:
                # Use pickle for simplicity, but torch.save might be safer for tensors
                pickle.dump(self.get_state_dict(), f)
            
            artifact = wandb.Artifact(name=name, type="model", description="TransformerModel state dict")
            artifact.add_file(temp_file_path)
            
            logged_artifact = wandb.log_artifact(artifact)
            print(f"Logging artifact '{name}' to Wandb...")
            logged_artifact.wait()
            print("Artifact logging complete.")
            
        except Exception as e:
             print(f"Error saving model artifact to Wandb: {e}")
        finally:
            shutil.rmtree(temp_dir)


    # --- Inner nn.Module Definition ---
    class TransformerModule(nn.Module):
        def __init__(
            self,
            input_dim: int,      # Dimension of input features (e.g., MFCC count)
            output_dim: int,     # Number of output classes
            d_model: int,        # Embedding dimension
            nhead: int,          # Number of attention heads
            num_encoder_layers: int,
            num_decoder_layers: int, # Not used currently
            dim_feedforward: int,
            dropout: float,
            max_seq_len: int     # Max sequence length for positional encoding
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.max_seq_len = max_seq_len

            # Input Embedding: Linear layer to project input features to d_model
            self.input_embedding = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

            # Standard Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                batch_first=True # Important: Input format is (Batch, Seq, Feature)
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

            # Output Layer: Map encoder output to class scores
            # Option 1: Use the output of the [CLS] token (if added)
            # Option 2: Average pool the output sequence
            # Option 3: Take the output of the first token (simpler)
            # For now, let's average pool the sequence output.
            self.output_layer = nn.Linear(d_model, output_dim)
            self.softmax = nn.Softmax(dim=1) # Apply softmax for probability distribution


        def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
            """
            Args:
                src: Input sequence, shape (Batch, Seq, Feature)
                src_key_padding_mask: Mask for padding tokens, shape (Batch, Seq). 
                                      True indicates valid token, False indicates padding.
                                      Note: PyTorch Transformer expects True for padded positions. We need to invert.
            
            Returns:
                Output tensor, shape (Batch, NumClasses)
            """
            # print(f"Module Input shape: {src.shape}")
            # print(f"Module Mask shape: {src_key_padding_mask.shape}")
            
            # 1. Embed input features
            src_embedded = self.input_embedding(src) * math.sqrt(self.d_model) # Scale embedding
            # print(f"Embedded shape: {src_embedded.shape}") # B, S, D
            
            # 2. Add positional encoding
            # PositionalEncoding expects (Seq, Batch, Dim) by default if batch_first=False
            # Since our PE is not batch_first, we might need to permute or adjust PE.
            # Let's assume PositionalEncoding is adapted or we permute:
            # src_embedded = src_embedded.permute(1, 0, 2) # S, B, D
            # src_embedded = self.pos_encoder(src_embedded)
            # src_embedded = src_embedded.permute(1, 0, 2) # B, S, D 
            # --- Simpler approach if PE handles batch_first ---
            src_embedded = self.pos_encoder(src_embedded.permute(1, 0, 2)).permute(1, 0, 2) # Apply PE (S,B,D) -> (B,S,D)
            # print(f"PosEncoded shape: {src_embedded.shape}")

            # 3. Pass through Transformer Encoder
            # TransformerEncoderLayer expects src_key_padding_mask where True indicates positions to be IGNORED.
            # Our mask has True for valid tokens. We need to invert it.
            padding_mask = ~src_key_padding_mask 
            # print(f"Inverted Mask shape: {padding_mask.shape}")

            encoder_output = self.transformer_encoder(src_embedded, src_key_padding_mask=padding_mask)
            # print(f"Encoder Output shape: {encoder_output.shape}") # B, S, D

            # 4. Aggregate sequence output (Mean Pooling over sequence dim)
            # Mask the padded outputs before averaging.
            # Expand mask to match encoder_output dimensions (B, S, D)
            mask_expanded = src_key_padding_mask.unsqueeze(-1).expand_as(encoder_output)
            # Zero out padded values
            masked_output = encoder_output * mask_expanded
            # Sum valid tokens and divide by the number of valid tokens per sequence
            summed_output = masked_output.sum(dim=1) # B, D
            valid_tokens_count = src_key_padding_mask.sum(dim=1, keepdim=True) # B, 1
            # Avoid division by zero for sequences with no valid tokens (shouldn't happen with proper input)
            valid_tokens_count = torch.max(valid_tokens_count, torch.tensor(1.0, device=src.device))
            mean_pooled_output = summed_output / valid_tokens_count # B, D
            # print(f"Mean Pooled shape: {mean_pooled_output.shape}")

            # 5. Final Linear Layer + Softmax
            output = self.output_layer(mean_pooled_output) # B, C
            output = self.softmax(output) # Apply Softmax
            # print(f"Final Output shape: {output.shape}")

            return output

