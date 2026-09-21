import random
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@torch.no_grad()
def extract_features(backbone, loader, device, max_samples=None):
    backbone.eval()
    feats, labels = [], []
    for imgs, lbls, _ in loader:
        imgs = imgs.to(device)
        f = backbone(imgs).cpu().numpy()
        feats.append(f)
        labels.append(lbls.numpy())
        if max_samples and sum(len(x) for x in feats) >= max_samples:
            break
    if not feats:
        raise ValueError("Cannot extract features from an empty data loader.")
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    if max_samples:
        feats = feats[:max_samples]
        labels = labels[:max_samples]
    return feats, labels

def compute_domain_separability(backbone, source_val_loaders, target_loader, device, seed=6304):
    """
    Collects equal numbers of source validation and target features,
    splits 70/30, and trains a LogisticRegression(C=1.0) domain discriminator.
    Chance = 50.0%.
    """
    # 1. Collect source features pooled evenly across source val splits
    source_features = []
    for ldr in source_val_loaders.values():
        f, _ = extract_features(backbone, ldr, device)
        source_features.append(f)
    if not source_features or any(len(f) == 0 for f in source_features):
        raise ValueError("Domain separability requires non-empty source validation loaders.")
    rng = np.random.RandomState(seed)
    per_domain = min(len(f) for f in source_features)
    src_feats = np.concatenate([
        f[rng.choice(len(f), size=per_domain, replace=False)]
        for f in source_features
    ], axis=0)

    # 2. Collect target features
    tgt_feats, _ = extract_features(backbone, target_loader, device)

    # 3. Balance sample counts
    n = min(len(src_feats), len(tgt_feats))
    idx_s = rng.choice(len(src_feats), size=n, replace=False)
    idx_t = rng.choice(len(tgt_feats), size=n, replace=False)

    X = np.concatenate([src_feats[idx_s], tgt_feats[idx_t]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)], axis=0)

    if n < 2:
        raise ValueError("Domain separability requires at least two source and target samples.")

    # 4. Stratified 70/30 split
    indices = np.arange(2 * n)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=y
    )

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    separability_score = accuracy_score(y[test_idx], preds) * 100.0

    return float(separability_score)
