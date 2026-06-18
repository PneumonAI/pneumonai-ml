from torch import nn
from torchvision import models

def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model type: {arch}")
    model.fc = nn.Linear(512, num_classes)
    return model