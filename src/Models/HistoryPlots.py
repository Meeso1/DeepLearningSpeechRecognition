import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from Models.TrainingHistory import TrainingHistory


def plot_loss_history(history: TrainingHistory, ax: plt.Axes | None = None):
    made_new_ax = False
    if ax is None:
        made_new_ax = True
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(history.train_loss, label='Train Loss')
    ax.plot(history.val_loss, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    if made_new_ax:
        plt.show()
        

def plot_accuracy_history(history: TrainingHistory, ax: plt.Axes | None = None):
    made_new_ax = False
    if ax is None:
        made_new_ax = True
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(history.train_accuracy, label='Train Accuracy')
    ax.plot(history.val_accuracy, label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy [%]')
    ax.legend()

    if made_new_ax:
        plt.show()

def plot_confusion_matrix(true_labels: list[int], predicted_labels: list[int], labels: list[str]):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalise
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
