import torch
import torch.nn as nn

class SourceOnlyModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x_s, y_s, **kwargs):
        feat_s = self.backbone(x_s)
        logits_s = self.classifier(feat_s)
        loss = self.criterion(logits_s, y_s)
        return loss, {"cls_loss": loss.item(), "total_loss": loss.item()}

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)