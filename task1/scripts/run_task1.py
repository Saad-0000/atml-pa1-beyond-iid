import argparse
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from task1.analysis.representation import (
    compute_cosine_stability,
    plot_umap_projections,
)
from task1.configs import configs
from task1.analysis.evaluate_bias import run_bias_evaluation
from task1.analysis.evaluate_cue_conflicts import run_cue_conflict_evaluation
from task1.data.transforms import (
    apply_cardinal_translation,
    apply_grayscale,
    apply_hue_rotation,
    apply_patch_shuffle,
)
from task1.models.backbones import (
    OpenCLIPWrapper,
    ResNet50Wrapper,
    ViTB16Wrapper,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms.v2 as T


class LinearProbe(nn.Module):

    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_head(
    train_loader,
    val_loader,
    backbone,
    in_dim,
    device,
    max_epochs,
    smoke_test=False,
):
    head = LinearProbe(in_dim, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=configs.LEARNING_RATE,
        weight_decay=configs.WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    best_weights = None

    for epoch in range(max_epochs):
        head.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            if smoke_test and batch_idx >= configs.MAX_TRAIN_BATCHES:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = backbone(x)
            optimizer.zero_grad()
            out = head(feats)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation check
        head.eval()
        all_preds, all_y = [], []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if smoke_test and batch_idx >= configs.MAX_VAL_BATCHES:
                    break
                x = x.to(device)
                feats = backbone(x)
                preds = head(feats).argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_y.extend(y.numpy())

        val_acc = (
            accuracy_score(all_y, all_preds) if len(all_y) > 0 else 0.0
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_weights = head.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= configs.EARLY_STOPPING_PATIENCE:
                break

    if best_weights is not None:
        head.load_state_dict(best_weights)
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=configs.DATA_ROOT)
    parser.add_argument(
        "--splits_file", type=str, default=configs.SPLITS_FILE
    )
    parser.add_argument(
        "--output_dir", type=str, default=configs.OUTPUT_DIR
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=configs.SMOKE_TEST,
        help="Quick debug execution on small batches",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.splits_file, "r") as f:
        splits = json.load(f)

    base_tf = T.Compose(
        [
            T.ToImage(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    full_train = STL10(
        root=args.data_dir, split="train", download=False, transform=base_tf
    )
    full_test = STL10(
        root=args.data_dir, split="test", download=False, transform=base_tf
    )

    train_ds = Subset(full_train, splits["train_indices"])
    val_ds = Subset(full_train, splits["val_indices"])
    eval_ds = Subset(full_test, splits["eval_test_indices"])

    batch_size = (
        16 if args.smoke_test else configs.BATCH_SIZE
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    models_dict = {
        "ResNet-50": ResNet50Wrapper().to(device),
        "ViT-B/16": ViTB16Wrapper().to(device),
        "CLIP-ViT-B/32": OpenCLIPWrapper(device=device),
    }

    norm_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    results = {}
    trained_heads = {}  # Store heads here so they can be passed to evaluate_bias

    for name, backbone in models_dict.items():
        print(f"--- Processing {name} ---")
        max_epochs = 1 if args.smoke_test else configs.MAX_EPOCHS
        head = train_head(
            train_loader,
            val_loader,
            backbone,
            backbone.feature_dim,
            device,
            max_epochs,
            smoke_test=args.smoke_test,
        )
        
        trained_heads[name] = head  # Save the trained linear probe

        clean_preds, clean_conf, targets, clean_feats = [], [], [], []
        head.eval()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(eval_loader):
                if args.smoke_test and batch_idx >= configs.MAX_VAL_BATCHES:
                    break
                x_norm = norm_transform(x).to(device)
                f = backbone(x_norm)
                logits = head(f)
                probs = torch.softmax(logits, dim=-1)
                conf, p = probs.max(dim=-1)

                clean_preds.extend(p.cpu().numpy())
                clean_conf.extend(conf.cpu().numpy())
                targets.extend(y.numpy())
                clean_feats.append(f.cpu())

        clean_feats = torch.cat(clean_feats, dim=0)
        clean_acc = accuracy_score(targets, clean_preds)
        clean_f1 = f1_score(targets, clean_preds, average="macro")

        # Grayscale intervention
        gray_preds, gray_feats = [], []
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(eval_loader):
                if args.smoke_test and batch_idx >= configs.MAX_VAL_BATCHES:
                    break
                x_gray = apply_grayscale(x)
                x_norm = norm_transform(x_gray).to(device)
                f = backbone(x_norm)
                p = head(f).argmax(dim=-1)
                gray_preds.extend(p.cpu().numpy())
                gray_feats.append(f.cpu())

        gray_feats = torch.cat(gray_feats, dim=0)
        gray_acc = accuracy_score(targets, gray_preds)
        gray_consistency = np.mean(np.array(gray_preds) == np.array(clean_preds))
        gray_stability = compute_cosine_stability(clean_feats, gray_feats)

        results[name] = {
            "clean_acc": clean_acc,
            "clean_macro_f1": clean_f1,
            "gray_acc_drop": clean_acc - gray_acc,
            "gray_consistency": gray_consistency,
            "gray_cosine_stability": gray_stability,
        }

    # Save the basic baseline metrics
    with open(os.path.join(args.output_dir, "task1_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Extended bias evaluation (Color, Translation, Patch Shuffle)
    print("Running extended bias evaluations (Translation, Patch Shuffle, etc.)...")
    bias_results = run_bias_evaluation(
        heads_dict=trained_heads,  # Now correctly passing the saved linear probes
        splits_file=args.splits_file,
        data_dir=args.data_dir,
        smoke_test=args.smoke_test,
    )

    with open(
        os.path.join(args.output_dir, "bias_evaluation_results.json"), "w"
    ) as f:
        json.dump(bias_results, f, indent=2)

    # Extended cue conflict evaluation
    print("Evaluating shape vs. texture on cue-conflict images...")
    conflict_results = run_cue_conflict_evaluation(
        heads_dict=trained_heads,
        splits_file=args.splits_file
    )
    
    if conflict_results:
        with open(os.path.join(args.output_dir, "cue_conflict_results.json"), "w") as f:
            json.dump(conflict_results, f, indent=2)

    print("Task 1 completed. Results saved.")


if __name__ == "__main__":
    main()