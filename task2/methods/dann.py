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
        self.grl_max = float(grl_max)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

    def _get_alpha(self, progress):
        progress = max(0.0, min(1.0, float(progress)))

        alpha = self.grl_max * (
            2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        )

        return float(alpha)

    def train_step(self, x_s, y_s, x_t, progress, **kwargs):

        alpha = self._get_alpha(progress)

        # -----------------------------------------------------
        # Feature extraction
        # -----------------------------------------------------

        feat_s = self.backbone(x_s)
        feat_t = self.backbone(x_t)

        # Monitor feature magnitude
        feat_s_norm = feat_s.detach().norm(dim=1).mean().item()
        feat_t_norm = feat_t.detach().norm(dim=1).mean().item()

        feat_s_max = feat_s.detach().abs().max().item()
        feat_t_max = feat_t.detach().abs().max().item()

        # -----------------------------------------------------
        # Source classification
        # -----------------------------------------------------

        logits_s = self.classifier(feat_s)

        cls_loss = self.cls_criterion(
            logits_s,
            y_s,
        )

        # -----------------------------------------------------
        # Domain classification
        # -----------------------------------------------------

        feat_combined = torch.cat(
            [feat_s, feat_t],
            dim=0,
        )

        reversed_feat = self.grl(
            feat_combined,
            alpha,
        )

        domain_preds = self.discriminator(
            reversed_feat
        )

        # Monitor discriminator logits
        domain_logit_mean = (
            domain_preds.detach().abs().mean().item()
        )

        domain_logit_max = (
            domain_preds.detach().abs().max().item()
        )

        domain_labels = torch.cat(
            [
                torch.zeros(
                    feat_s.size(0),
                    dtype=torch.long,
                    device=feat_s.device,
                ),
                torch.ones(
                    feat_t.size(0),
                    dtype=torch.long,
                    device=feat_t.device,
                ),
            ],
            dim=0,
        )

        domain_loss = self.domain_criterion(
            domain_preds,
            domain_labels,
        )

        # -----------------------------------------------------
        # Total loss
        # -----------------------------------------------------

        total_loss = cls_loss + domain_loss

        # -----------------------------------------------------
        # Numerical checks
        # -----------------------------------------------------

        if not torch.isfinite(cls_loss):
            raise RuntimeError(
                f"Classification loss became non-finite: "
                f"{cls_loss.item()}"
            )

        if not torch.isfinite(domain_loss):
            raise RuntimeError(
                f"Domain loss became non-finite: "
                f"{domain_loss.item()}"
            )

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Total loss became non-finite: "
                f"{total_loss.item()}"
            )

        return total_loss, {
            "cls_loss": float(cls_loss.detach().item()),
            "domain_loss": float(domain_loss.detach().item()),
            "alpha": float(alpha),
            "total_loss": float(total_loss.detach().item()),

            "feat_s_norm": feat_s_norm,
            "feat_t_norm": feat_t_norm,
            "feat_s_max": feat_s_max,
            "feat_t_max": feat_t_max,

            "domain_logit_mean": domain_logit_mean,
            "domain_logit_max": domain_logit_max,
        }

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)