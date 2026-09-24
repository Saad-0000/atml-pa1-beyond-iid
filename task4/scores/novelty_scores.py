import torch
import numpy as np

def compute_msp(logits):
    """u_MSP(x) = 1 - max(softmax(logits))[cite: 2]"""
    probs = torch.softmax(logits, dim=1)
    return 1.0 - torch.max(probs, dim=1)[0].cpu().numpy()

def compute_energy(logits):
    """u_Energy(x) = -log(sum(exp(logits)))[cite: 2]"""
    return -torch.logsumexp(logits, dim=1).cpu().numpy()

def compute_mls(logits):
    """u_MLS(x) = -max(logits)[cite: 2]"""
    return -torch.max(logits, dim=1)[0].cpu().numpy()

def fit_mahalanobis(features, labels, num_classes=10):
    means = []
    cov_sum = torch.zeros((features.size(1), features.size(1)), device=features.device)
    for c in range(num_classes):
        class_feats = features[labels == c]
        mean_c = class_feats.mean(dim=0)
        means.append(mean_c)
        centered = class_feats - mean_c
        cov_sum += torch.mm(centered.T, centered)
        
    cov = cov_sum / features.size(0)
    cov += torch.eye(features.size(1), device=features.device) * 1e-6 # Add 1e-6 to diagonal[cite: 2]
    precision = torch.inverse(cov)
    return torch.stack(means), precision

def compute_mahalanobis(features, means, precision):
    """u_Mah(x) = min_c (f - u_c)^T \Sigma^-1 (f - u_c)[cite: 2]"""
    scores = []
    for f in features:
        dists = []
        for u_c in means:
            diff = (f - u_c).unsqueeze(0)
            dist = torch.mm(torch.mm(diff, precision), diff.T)
            dists.append(dist.item())
        scores.append(min(dists))
    return np.array(scores)