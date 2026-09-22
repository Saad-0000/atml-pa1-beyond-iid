import math

import torch
import torch.nn as nn

from task2.models.domain_discriminator import GradientReversalLayer


class DANNModel(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN).

    Source data:
        x_s, y_s

    Target data:
        x_t

    The backbone produces features for both domains.

    Source features are used for:
        1. Class prediction
        2. Domain prediction

    Target features are used for:
        1. Domain prediction

    The Gradient Reversal Layer reverses the domain-loss gradient
    before it reaches the feature extractor.
    """

    def __init__(
        self,
        backbone,
        classifier,
        discriminator,
        grl_max=1.0,
    ):
        super().__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.discriminator = discriminator

        self.grl = GradientReversalLayer()

        self.grl_max = float(grl_max)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

    def _get_alpha(self, progress):
        """
        DANN GRL schedule:

            alpha(p) =
                grl_max * [2 / (1 + exp(-10p)) - 1]

        where p is training progress in [0, 1].
        """

        # Keep progress safely inside [0, 1].
        progress = max(0.0, min(1.0, float(progress)))

        alpha = self.grl_max * (
            2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        )

        return float(alpha)

    def train_step(self, x_s, y_s, x_t, progress, **kwargs):

        # ---------------------------------------------------------
        # 1. Compute GRL coefficient
        # ---------------------------------------------------------

        alpha = self._get_alpha(progress)

        # ---------------------------------------------------------
        # 2. Extract source and target features
        # ---------------------------------------------------------

        feat_s = self.backbone(x_s)
        feat_t = self.backbone(x_t)

        # ---------------------------------------------------------
        # 3. Source classification loss
        # ---------------------------------------------------------

        logits_s = self.classifier(feat_s)

        cls_loss = self.cls_criterion(
            logits_s,
            y_s,
        )

        # ---------------------------------------------------------
        # 4. Domain classification
        # ---------------------------------------------------------

        feat_combined = torch.cat(
            [feat_s, feat_t],
            dim=0,
        )

        # Gradient reversal:
        # discriminator receives normal features,
        # backbone receives reversed domain gradient.
        reversed_feat = self.grl(
            feat_combined,
            alpha,
        )

        domain_preds = self.discriminator(
            reversed_feat
        )

        # Source = 0
        # Target = 1
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

        # ---------------------------------------------------------
        # 5. Total DANN objective
        # ---------------------------------------------------------

        # IMPORTANT:
        # Do NOT subtract domain_loss here.
        #
        # The Gradient Reversal Layer already reverses the
        # domain gradient going into the feature extractor.

        total_loss = cls_loss + domain_loss

        # ---------------------------------------------------------
        # 6. Numerical safety checks
        # ---------------------------------------------------------

        if not torch.isfinite(cls_loss):
            raise RuntimeError(
                f"Non-finite classification loss detected: "
                f"{cls_loss.item()}"
            )

        if not torch.isfinite(domain_loss):
            raise RuntimeError(
                f"Non-finite domain loss detected: "
                f"{domain_loss.item()}"
            )

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite total loss detected: "
                f"{total_loss.item()}"
            )

        return total_loss, {
            "cls_loss": float(cls_loss.detach().item()),
            "domain_loss": float(domain_loss.detach().item()),
            "alpha": float(alpha),
            "total_loss": float(total_loss.detach().item()),
        }

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)