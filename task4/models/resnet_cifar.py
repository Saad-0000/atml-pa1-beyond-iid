import torch
import torch.nn as nn
from torchvision.models import resnet18

class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = resnet18(weights=None)
        # Replace 7x7 stride-2 conv with 3x3 stride-1 and remove max-pool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
        self.features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool
        )
        self.fc = nn.Linear(model.fc.in_features, num_classes)
        self.feature_dim = model.fc.in_features

    def forward(self, x, return_feature=False):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        if return_feature:
            return logits, feat
        return logits
        
    def get_proser_layers(self):
        """Exposes network split for Manifold Mixup between layer2 and layer3[cite: 2]."""
        model = resnet18(weights=None)
        pre_mixup = nn.Sequential(
            self.features[0], self.features[1], self.features[2], self.features[3],
            self.features[4], self.features[5] # up to layer2
        )
        post_mixup = nn.Sequential(
            self.features[6], self.features[7], self.features[8] # layer3 to avgpool
        )
        return pre_mixup, post_mixup