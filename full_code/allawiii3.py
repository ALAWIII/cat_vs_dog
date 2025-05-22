from torchvision import models
from torch.cuda.amp import GradScaler
import torch.optim as optim
import torch.nn as nn
from .config import DEVICE, LEARNING_RATE, EPOCHS


def load_pretrained_architecture() -> models.EfficientNet:
    model = models.efficientnet_b0(pretrained=True)
    in_features = (
        model.classifier[1].in_features
        if isinstance(model.classifier[1], nn.Linear)
        else model.classifier[-1].in_features
    )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, 2)
    )
    model = model.to(DEVICE)
    return model


def loss(model: models.EfficientNet):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()  # for mixed precision
    return criterion, optimizer, scheduler, scaler
