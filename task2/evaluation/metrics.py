import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if all_targets.size == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    return acc, macro_f1
