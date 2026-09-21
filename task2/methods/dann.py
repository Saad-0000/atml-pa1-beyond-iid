import math
import torch
import torch.nn as nn
from task2.models.domain_discriminator import GradientReversalLayer

class DANNModel(nn.Module):
    def __init__(self, backbone, classifier, discriminator, grl_max=1.0):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.discriminator = discriminator
        self.grl = GradientReversalLayer()
        self.grl_max = grl_max
        self.cls_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

    def train_step(self, x_s, y_s, x_t, progress, **kwargs):
        # Schedule: alpha(p) = (2 / (1 + exp(-10*p))) - 1
        alpha = self.grl_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)

        feat_s = self.backbone(x_s)
        feat_t = self.backbone(x_t)

        logits_s = self.classifier(feat_s)
        cls_loss = self.cls_criterion(logits_s, y_s)

        # Domain classification
        feat_combined = torch.cat([feat_s, feat_t], dim=0)
        reversed_feat = self.grl(feat_combined, alpha)
        domain_preds = self.discriminator(reversed_feat)

        # Domain labels: Source=0, Target=1
        domain_labels = torch.cat([
            torch.zeros(feat_s.size(0), dtype=torch.long, device=x_s.device),
            torch.ones(feat_t.size(0), dtype=torch.long, device=x_t.device)
        ], dim=0)

        domain_loss = self.domain_criterion(domain_preds, domain_labels)
        total_loss = cls_loss + domain_loss

        return total_loss, {
            "cls_loss": cls_loss.item(),
            "domain_loss": domain_loss.item(),
            "alpha": float(alpha),
            "total_loss": total_loss.item()
        }

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)
