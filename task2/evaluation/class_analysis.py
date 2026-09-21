import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from shared.pacs import PACS_CLASSES

@torch.no_grad()
def evaluate_per_class_and_confusions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    cm = confusion_matrix(all_targets, all_preds, labels=range(len(PACS_CLASSES)))
    per_class_acc = (cm.diagonal() / np.maximum(cm.sum(axis=1), 1)) * 100.0

    class_metrics = {}
    for i, cname in enumerate(PACS_CLASSES):
        # Find primary confusion class (excluding ground truth)
        row = cm[i].copy()
        row[i] = -1
        top_confusion_idx = np.argmax(row)
        top_conf_count = row[top_confusion_idx]

        class_metrics[cname] = {
            "accuracy": float(per_class_acc[i]),
            "primary_confusion": PACS_CLASSES[top_confusion_idx],
            "confusion_count": int(top_conf_count)
        }

    return class_metrics, cm