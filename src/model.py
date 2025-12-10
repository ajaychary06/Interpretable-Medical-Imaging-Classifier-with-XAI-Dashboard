# src/model.py
import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=2, pretrained=False):
    """
    Returns a ResNet18 modified for `num_classes`.
    Set pretrained=True if you want to download ImageNet weights (requires internet).
    """
    model = models.resnet18(weights=None) if not pretrained else models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
