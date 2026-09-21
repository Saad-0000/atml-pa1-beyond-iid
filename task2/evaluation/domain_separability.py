import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def extract_features(backbone, loader, device, max_samples=None):
    backbone.eval()

    feats = []
    labels = []
    n_collected = 0

    for imgs, lbls, _ in loader:
        imgs = imgs.to(device)

        f = backbone(imgs).cpu().numpy()

        feats.append(f)
        labels.append(lbls.numpy())

        n_collected += len(lbls)

        if max_samples is not None and n_collected >= max_samples:
            break

    if not feats:
        raise ValueError(
            "Cannot extract features from an empty data loader."
        )

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    if max_samples is not None:
        feats = feats[:max_samples]
        labels = labels[:max_samples]

    return feats, labels


def compute_domain_separability(
    backbone,
    source_val_loaders,
    target_loader,
    device,
    seed=6304
):
    """
    Measures how easily a classifier can distinguish source
    features from target features.

    50% ~= domain-invariant features
    100% ~= highly domain-separable features

    The domain classifier is evaluated on a held-out test split.
    """

    # ---------------------------------------------------------
    # 1. Extract source validation features
    # ---------------------------------------------------------

    source_features = []

    for loader in source_val_loaders.values():

        features, _ = extract_features(
            backbone,
            loader,
            device
        )

        source_features.append(features)

    if not source_features:
        raise ValueError(
            "No source features available."
        )

    # Balance the three source domains
    rng = np.random.RandomState(seed)

    per_domain = min(
        len(features)
        for features in source_features
    )

    balanced_source = []

    for features in source_features:

        indices = rng.choice(
            len(features),
            size=per_domain,
            replace=False
        )

        balanced_source.append(
            features[indices]
        )

    source_features = np.concatenate(
        balanced_source,
        axis=0
    )

    # ---------------------------------------------------------
    # 2. Extract target features
    # ---------------------------------------------------------

    target_features, _ = extract_features(
        backbone,
        target_loader,
        device
    )

    # ---------------------------------------------------------
    # 3. Equalize source / target sample counts
    # ---------------------------------------------------------

    n = min(
        len(source_features),
        len(target_features)
    )

    if n < 10:
        raise ValueError(
            "Too few samples for domain separability."
        )

    source_idx = rng.choice(
        len(source_features),
        size=n,
        replace=False
    )

    target_idx = rng.choice(
        len(target_features),
        size=n,
        replace=False
    )

    source_features = source_features[source_idx]
    target_features = target_features[target_idx]

    X = np.concatenate(
        [source_features, target_features],
        axis=0
    )

    y = np.concatenate(
        [
            np.zeros(n, dtype=np.int64),
            np.ones(n, dtype=np.int64)
        ]
    )

    # ---------------------------------------------------------
    # 4. Train/test split
    # ---------------------------------------------------------

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.30,
        random_state=seed,
        stratify=y
    )

    # ---------------------------------------------------------
    # 5. Standardized Logistic Regression
    # ---------------------------------------------------------

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=3000,
            random_state=seed
        )
    )

    clf.fit(
        X[train_idx],
        y[train_idx]
    )

    predictions = clf.predict(
        X[test_idx]
    )

    separability = (
        accuracy_score(
            y[test_idx],
            predictions
        ) * 100.0
    )

    return float(separability)