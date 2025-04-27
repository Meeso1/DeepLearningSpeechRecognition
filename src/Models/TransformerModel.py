from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import math

from Models.ModelBase import ModelBase
from torch.utils.data import DataLoader, TensorDataset
from Models.TrainingHistory import TrainingHistory
from Models.RandomUndersampler import RandomUndersampler # Keep for potential future use, though less common for transformers
from Models.WandbDetails import WandbDetails


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
        embedding_dimension: int = 512,
        num_attention_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        lr_decay: float = 0.0,
        beta_1: float = 0.9,
        beta_2: float = 0.98,
        eps: float = 1e-9,
        print_every: int | None = 1,
        validate_every: int = 1,
        wandb_details: WandbDetails | None = None,
        seed: int = 42
    ) -> None:
        self.classes: list[str] = classes
        self.num_classes: int = len(classes)
        self.labels_index: dict[str, int] = {v: i for i, v in enumerate(classes)}
        
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.learning_rate: float = learning_rate
        self.beta_1: float = beta_1
        self.beta_2: float = beta_2
        self.eps: float = eps
        self.lr_decay: float = lr_decay

        self.wandb_details: WandbDetails | None = wandb_details
        self.quiet: bool = print_every is None
        self.print_every: int = print_every if print_every is not None else 1
        self.validate_every: int = validate_every

        self.input_shape: tuple[int, int] | None = None
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.lr_scheduler: optim.lr_scheduler.LambdaLR | None = None
        self.criterion = nn.BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.history: TrainingHistory | None = None

        self.set_random_seed(seed)

    def _initialize_model(self, input_shape: tuple[int, int]) -> None:
        """Initialize stuff that depends on input shape."""
        self.input_shape = input_shape
        self.model = self.TransformerModule(
            input_dim=self.input_shape[1],
            output_dim=self.num_classes,
            d_model=self.embedding_dimension,
            nhead=self.num_attention_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_len=self.input_shape[0]
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            eps=self.eps
        )

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (1 - self.lr_decay) ** epoch if self.lr_decay > 0 else 1.0
        )

    def _generate_padding_mask(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Generate padding mask assuming zero vectors are padding.
        
        Args:
            batch_tensor: Input tensor of shape (B, S, F).

        Returns:
            mask: Boolean tensor of shape (B, S) where True indicates a valid token.
        """
        # Check if the sum of absolute values across the feature dimension is zero
        # Add a small epsilon for numerical stability if needed, though comparing to 0 should be fine for exact zeros
        mask = torch.abs(batch_tensor).sum(dim=2) != 0 
        return mask

    def _validate_x(self, X: list[np.ndarray]) -> None:
        """Validate X for shape consistency (sequence length and feature dimension)."""
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
            raise ValueError(f"Number of samples in X ({len(X)}) does not match y ({len(y)})")

        # Check that y contains valid class indices
        if len(y) > 0 and (np.max(y) >= self.num_classes or np.min(y) < 0):
            raise ValueError(f"y contains invalid class indices. Expected range [0, {self.num_classes-1}], got range [{np.min(y)}, {np.max(y)}]")

    def _validate_data_and_make_loader(
        self, 
        data: list[tuple[np.ndarray, np.ndarray]], 
        batch_size: int, 
        undersample: bool = False,
        shuffle: bool = True
    ) -> DataLoader:
        """Validate (X, y) pair and make DataLoader for it"""

        X, y = zip(*data)
        X, y = list(X), np.array(y)
        
        self._validate_x_and_y(X, y)

        X_np = np.array(X, dtype=np.float32) 
        y_one_hot = self._to_one_hot(y)

        X_tensor = torch.from_numpy(X_np).to(self.device)
        y_tensor = torch.from_numpy(y_one_hot).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        sampler = RandomUndersampler(torch.from_numpy(y).to(self.device)) if undersample else None
        
        # If using sampler, shuffle must be False
        use_shuffle = shuffle and (sampler is None) 
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=use_shuffle, sampler=sampler)

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
            self._initialize_model(train_data[0][0].shape)

        train_loader = self._validate_data_and_make_loader(train_data, batch_size, shuffle=True)
        val_loader = self._validate_data_and_make_loader(val_data, batch_size, shuffle=False) \
            if val_data is not None else None
        
        if self.history is None:
            self.history = TrainingHistory()

        if self.wandb_details is not None and self.wandb_details.init_project:
            wandb.init(
                project=self.wandb_details.project,
                name=self.wandb_details.experiment_name,
                config=self.get_config_for_wandb(),
                settings=wandb.Settings(silent=True)
            )

        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for _, (inputs, targets_one_hot) in enumerate(train_loader):
                # Inputs: (B, S, F), Targets: (B, C)
                inputs = inputs.to(self.device)
                targets_one_hot = targets_one_hot.to(self.device)
                
                # Generate mask from pre-padded data
                src_key_padding_mask = self._generate_padding_mask(inputs) # (B, S), True=valid

                self.optimizer.zero_grad()

                outputs = self.model(inputs, src_key_padding_mask) # Pass mask
                
                loss = self.criterion(outputs, targets_one_hot)

                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_train_loss += loss.detach().item() * inputs.size(0) 
                _, predicted = outputs.max(1)
                total_train += targets_one_hot.size(0)
                correct_train += predicted.eq(targets_one_hot.argmax(dim=1)).sum().item()

            epoch_train_loss /= len(train_loader.dataset)
            epoch_train_accuracy = 100.0 * correct_train / total_train
            self.history.train_loss.append(epoch_train_loss)
            self.history.train_accuracy.append(epoch_train_accuracy)


            epoch_val_loss, epoch_val_accuracy = self._perform_validation(val_loader, epoch)
            
            self.history.val_loss.append(epoch_val_loss)
            self.history.val_accuracy.append(epoch_val_accuracy)

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

    def _perform_validation(self, val_loader: DataLoader | None, epoch: int) -> tuple[float, float] | None:
        """Perform validation using a DataLoader"""
        if val_loader is None or (epoch + 1) % self.validate_every != 0:
            return None, None

        self.model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets_one_hot in val_loader:
                # Inputs: (B, S, F), Targets: (B, C)
                inputs = inputs.to(self.device)
                targets_one_hot = targets_one_hot.to(self.device)
                
                # Generate mask from pre-padded data
                src_key_padding_mask = self._generate_padding_mask(inputs) # (B, S), True=valid

                outputs = self.model(inputs, src_key_padding_mask)
                loss = self.criterion(outputs, targets_one_hot)

                val_loss += loss.detach().item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets_one_hot.size(0)
                correct_val += predicted.eq(targets_one_hot.argmax(dim=1)).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
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
        self._validate_x(X)

        self.model.eval()
        all_predicted = []

        X_np = np.array(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_np).to(self.device)

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device) # DataLoader wraps tensor in a tuple/list
                
                # Generate mask from pre-padded data
                src_key_padding_mask = self._generate_padding_mask(inputs) # (B, S), True=valid
                
                outputs = self.model(inputs, src_key_padding_mask) # B, C
                _, predicted = torch.max(outputs, 1) # Get class index
                all_predicted.append(predicted.cpu())

        return torch.cat(all_predicted).numpy()

    def get_history(self) -> TrainingHistory:
        return self.history if self.history is not None else TrainingHistory()

    def get_state_dict(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("Model has not been initialized or trained yet")

        return {
            # Model params
            'input_shape': self.input_shape,
            'embedding_dimension': self.embedding_dimension,
            'num_attention_heads': self.num_attention_heads,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            # Optimizer params
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'eps': self.eps,
            # Training config
            'classes': self.classes,
            'history': self.history,
            'wandb_details': self.wandb_details,
            'print_every': self.print_every if not self.quiet else None,
            'validate_every': self.validate_every,
            # Model state
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        
    def get_config_for_wandb(self) -> dict[str, Any]:
        return {
            # Model
            'embedding_dimension': self.embedding_dimension,
            'num_attention_heads': self.num_attention_heads,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            # Optimizer
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'eps': self.eps
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "TransformerModel":
        model = TransformerModel(
            classes=state_dict['classes'],
            embedding_dimension=state_dict['embedding_dimension'],
            num_attention_heads=state_dict['num_attention_heads'],
            num_encoder_layers=state_dict['num_encoder_layers'],
            num_decoder_layers=state_dict['num_decoder_layers'],
            dim_feedforward=state_dict['dim_feedforward'],
            dropout=state_dict['dropout'],
             # Optimizer params
            learning_rate=state_dict['learning_rate'],
            lr_decay=state_dict.get('lr_decay', 0.0),
            beta_1=state_dict['beta_1'],
            beta_2=state_dict['beta_2'],
            eps=state_dict['eps'],
             # Training config
            print_every=state_dict.get('print_every'),
            validate_every=state_dict.get('validate_every', 1),
            wandb_details=state_dict.get('wandb_details') 
        )

        model._initialize_model(state_dict['input_shape'])
        
        model.model.load_state_dict(state_dict['model_state_dict'])
        model.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        model.lr_scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        model.history = state_dict.get('history')

        return model

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
            # Our mask (`src_key_padding_mask`) has True for valid tokens. We need to invert it.
            pytorch_padding_mask = ~src_key_padding_mask 
            # print(f"Inverted Mask shape: {pytorch_padding_mask.shape}") # True=ignore

            encoder_output = self.transformer_encoder(src_embedded, src_key_padding_mask=pytorch_padding_mask)
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
