import matplotlib.pyplot as plt
import numpy as np
from task1.configs import configs
import torch
import torch.nn.functional as F
import umap


def compute_cosine_stability(
    clean_feats: torch.Tensor, trans_feats: torch.Tensor
) -> float:
    """Computes I_T metric: cosine stability between paired representations."""
    sim = F.cosine_similarity(clean_feats, trans_feats, dim=-1)
    return sim.mean().item()


def plot_umap_projections(
    clean_feats: np.ndarray,
    trans_feats: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
    seed: int = configs.SEED,
):
    """Fits one UMAP projection to clean + transformed embeddings."""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
    combined = np.vstack([clean_feats, trans_feats])
    embedding = reducer.fit_transform(combined)

    n = len(clean_feats)
    clean_proj = embedding[:n]
    trans_proj = embedding[n:]

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i in range(10):
        idx = labels == i
        plt.scatter(
            clean_proj[idx, 0],
            clean_proj[idx, 1],
            c=[cmap(i)],
            marker="o",
            alpha=0.6,
            label=f"Class {i} Clean" if i == 0 else "",
        )
        plt.scatter(
            trans_proj[idx, 0],
            trans_proj[idx, 1],
            c=[cmap(i)],
            marker="^",
            alpha=0.6,
            label=f"Class {i} Transformed" if i == 0 else "",
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()