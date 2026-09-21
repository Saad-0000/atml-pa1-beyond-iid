import torch
import torch.nn as nn


def squared_mmd(source_features, target_features):
    """Biased squared MMD with an RBF kernel and a data-dependent bandwidth."""
    features = torch.cat([source_features, target_features], dim=0)
    squared_distances = torch.cdist(features, features).pow(2)

    # The median heuristic gives a stable bandwidth without a separate tuning
    # parameter.  Clamp it for degenerate batches (e.g. identical features).
    positive_distances = squared_distances[squared_distances > 0]
    bandwidth = (
        positive_distances.detach().median()
        if positive_distances.numel() > 0
        else features.new_tensor(1.0)
    ).clamp_min(1e-6)
    kernel = torch.exp(-squared_distances / (2.0 * bandwidth))

    n_source = source_features.size(0)
    k_ss = kernel[:n_source, :n_source]
    k_tt = kernel[n_source:, n_source:]
    k_st = kernel[:n_source, n_source:]
    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()


class DANModel(nn.Module):
    """Deep Adaptation Network: source classification plus MMD alignment."""

    def __init__(self, backbone, classifier, lambda_mmd=1.0):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.lambda_mmd = lambda_mmd
        self.cls_criterion = nn.CrossEntropyLoss()

    def train_step(self, x_s, y_s, x_t, **kwargs):
        feat_s = self.backbone(x_s)
        feat_t = self.backbone(x_t)
        logits_s = self.classifier(feat_s)
        cls_loss = self.cls_criterion(logits_s, y_s)
        mmd_loss = squared_mmd(feat_s, feat_t)
        total_loss = cls_loss + self.lambda_mmd * mmd_loss
        return total_loss, {
            "cls_loss": cls_loss.item(),
            "mmd_loss": mmd_loss.item(),
            "total_loss": total_loss.item(),
        }

    def forward(self, x):
        return self.classifier(self.backbone(x))
