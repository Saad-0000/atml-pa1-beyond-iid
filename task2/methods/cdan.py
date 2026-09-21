import math
import torch
import torch.nn as nn
from task2.models.domain_discriminator import GradientReversalLayer

class CDANModel(nn.Module):
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
        alpha = self.grl_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)

        feat_s = self.backbone(x_s)
        feat_t = self.backbone(x_t)

        logits_s = self.classifier(feat_s)
        logits_t = self.classifier(feat_t)
        cls_loss = self.cls_criterion(logits_s, y_s)

        prob_s = torch.softmax(logits_s, dim=1)
        prob_t = torch.softmax(logits_t, dim=1)

        # Multilinear Conditioning: g(x) = vec(f (x) p)
        # B x 512 x 1, B x 1 x 7 -> B x 512 x 7 -> B x 3584
        cond_s = torch.bmm(feat_s.unsqueeze(2), prob_s.unsqueeze(1)).view(feat_s.size(0), -1)
        cond_t = torch.bmm(feat_t.unsqueeze(2), prob_t.unsqueeze(1)).view(feat_t.size(0), -1)

        cond_combined = torch.cat([cond_s, cond_t], dim=0)
        reversed_cond = self.grl(cond_combined, alpha)
        domain_preds = self.discriminator(reversed_cond)

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
