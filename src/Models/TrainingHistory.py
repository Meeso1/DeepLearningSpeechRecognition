from dataclasses import dataclass


@dataclass
class TrainingHistory:
    train_loss: list[float] = None
    train_accuracy: list[float] = None
    val_loss: list[float] = None
    val_accuracy: list[float] = None

    # Needed because default values in python are class-specific, not instance-specific
    def __post_init__(self):
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
