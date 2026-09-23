import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FrozenBNResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained weights
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove ImageNet classifier

    def forward(self, x):
        return self.backbone(x)

    def train(self, mode=True):
        super().train(mode)
        # Freeze BatchNorm running means and variances
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

class ClassifierHead(nn.Module):
    def __init__(self, in_features=512, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.fc(x)