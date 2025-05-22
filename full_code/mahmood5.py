 # --- mahmood --- evaluation

from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import EfficientNet
import torch

from .config import DEVICE


def evaluate_model(
    model: EfficientNet, full_dataset: datasets.ImageFolder, val_loader: DataLoader
):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=full_dataset.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
