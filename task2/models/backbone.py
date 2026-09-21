import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        # Drop classification head to output 512-dim pooled features
        model.fc = nn.Identity()
        self.encoder = model

    def forward(self, x):
        return self.encoder(x)

def freeze_bn_running_stats(module: nn.Module):
    """
    Enforces evaluation mode on all BatchNorm layers.
    Freezes running_mean and running_var to ImageNet statistics
    while keeping gamma (weight) and beta (bias) trainable.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()