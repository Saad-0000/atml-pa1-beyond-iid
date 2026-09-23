import torch
import torch.nn as nn

def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, float('inf'))

def mmd_rbf(feat1, feat2, bandwidths=[0.5, 1.0, 2.0]):
    dist_xx = compute_pairwise_distances(feat1, feat1)
    dist_yy = compute_pairwise_distances(feat2, feat2)
    dist_xy = compute_pairwise_distances(feat1, feat2)

    # Median heuristic for bandwidth
    median_dist = torch.median(dist_xy[dist_xy > 0])
    if median_dist == 0:
        median_dist = 1.0

    mmd = 0.0
    for scale in bandwidths:
        gamma = 1.0 / (2.0 * (scale * median_dist))
        k_xx = torch.exp(-gamma * dist_xx).mean()
        k_yy = torch.exp(-gamma * dist_yy).mean()
        k_xy = torch.exp(-gamma * dist_xy).mean()
        mmd += k_xx + k_yy - 2 * k_xy
    return mmd

def compute_dan_dg_loss(features, domain_labels, lambda_dg=1.0):
    """Computes MMD across the 3 source domains (Photo, Art, Cartoon)."""
    unique_domains = torch.unique(domain_labels)
    if len(unique_domains) < 2:
        return torch.tensor(0.0).to(features.device)
    
    mmd_loss = 0.0
    pairs = 0
    for i in range(len(unique_domains)):
        for j in range(i + 1, len(unique_domains)):
            f_i = features[domain_labels == unique_domains[i]]
            f_j = features[domain_labels == unique_domains[j]]
            if len(f_i) > 0 and len(f_j) > 0:
                mmd_loss += mmd_rbf(f_i, f_j)
                pairs += 1
                
    return (lambda_dg / pairs) * mmd_loss if pairs > 0 else torch.tensor(0.0)