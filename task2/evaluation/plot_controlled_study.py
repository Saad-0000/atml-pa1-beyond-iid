import os
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from shared.pacs import PACSDataset, get_pacs_transforms
from shared.pacs_protocol import create_or_load_pacs_splits
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.methods.source_only import SourceOnlyModel
from task2.evaluation.metrics import evaluate_loader
from task2.evaluation.domain_separability import compute_domain_separability


def evaluate_and_plot_controlled_study(
    study_type,
    data_root,
    ckpt_dir,
    num_workers=0,
    save_dir="./task2/results",
    split_json="splits/pacs_sketch_seed6304.json",
    debug=False
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # 1. Setup Study Parameters
    if study_type == "dann":
        param_values = [0.25, 0.5, 1.0]
        x_label = "Maximum GRL Strength (α)"
        title = "DANN: Effect of Alignment Strength"

    elif study_type == "dan":
        param_values = [0.1, 1.0, 10.0]
        x_label = "MMD Penalty Weight (λ)"
        title = "DAN: Effect of Alignment Strength"

    else:
        raise ValueError("study_type must be 'dann' or 'dan'")

    # 2. Setup Data Loaders
    splits = create_or_load_pacs_splits(data_root, split_json)
    _, eval_tf = get_pacs_transforms()

    source_val_loaders = {}

    for d in ["photo", "art_painting", "cartoon"]:
        ds_val = PACSDataset(
            splits["sources"][d]["val"],
            transform=eval_tf
        )

        if debug:
            ds_val = Subset(
                ds_val,
                range(min(20, len(ds_val)))
            )

        source_val_loaders[d] = DataLoader(
            ds_val,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers
        )

    target_ds = PACSDataset(
        splits["target"]["sketch"],
        transform=eval_tf
    )

    if debug:
        target_ds = Subset(
            target_ds,
            range(min(20, len(target_ds)))
        )

    target_loader = DataLoader(
        target_ds,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers
    )

    # 3. Evaluate Checkpoints
    mean_source_accs = []
    target_accs = []
    separabilities = []

    for val in param_values:
        ckpt_path = os.path.join(
            ckpt_dir,
            f"{study_type}_val{val}_best.pt"
        )

        print(f"\nEvaluating: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Did you train it?"
            )

        backbone = ResNet18Backbone()
        classifier = ClassifierHead(512, 7)
        model = SourceOnlyModel(
            backbone,
            classifier
        ).to(device)

        state_dict = torch.load(
            ckpt_path,
            map_location=device
        )

        inference_state = {
            k: v
            for k, v in state_dict.items()
            if k in model.state_dict()
        }

        model.load_state_dict(
            inference_state,
            strict=True
        )

        model.eval()

        # Target Accuracy
        tgt_acc, _ = evaluate_loader(
            model,
            target_loader,
            device
        )

        target_accs.append(tgt_acc)

        # Source Accuracy
        src_accs = [
            evaluate_loader(model, ldr, device)[0]
            for ldr in source_val_loaders.values()
        ]

        mean_source_accs.append(
            np.mean(src_accs)
        )

        # Domain Separability
        sep = compute_domain_separability(
            model.backbone,
            source_val_loaders,
            target_loader,
            device
        )

        separabilities.append(sep)

    # 4. Generate Plot
    plt.figure(figsize=(6, 4.5))
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.plot(
        param_values,
        mean_source_accs,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="Mean Source Acc (%)"
    )

    plt.plot(
        param_values,
        target_accs,
        marker="s",
        linewidth=2,
        color="tab:green",
        label="Target Acc (%)"
    )

    plt.plot(
        param_values,
        separabilities,
        marker="^",
        linewidth=2,
        color="tab:red",
        linestyle="--",
        label="Domain Separability (%)"
    )

    plt.title(
        title,
        fontsize=12,
        fontweight="bold"
    )

    plt.xlabel(
        x_label,
        fontsize=11
    )

    plt.ylabel(
        "Percentage (%)",
        fontsize=11
    )

    if study_type == "dan":
        plt.xscale("log")
        plt.xticks(
            param_values,
            [str(v) for v in param_values]
        )
    else:
        plt.xticks(param_values)

    plt.ylim(0, 100)
    plt.legend(loc="best")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    suffix = "_debug" if debug else ""

    save_path = os.path.join(
        save_dir,
        f"{study_type}_controlled_study{suffix}.pdf"
    )

    plt.savefig(
        save_path,
        format="pdf",
        dpi=300
    )

    print(
        f"\nPlot saved successfully to: {save_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--study_type",
        type=str,
        required=True,
        choices=["dan", "dann"]
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/PACS"
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./task2/results/checkpoints"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./task2/results"
    )

    parser.add_argument(
        "--split_json",
        type=str,
        default="splits/pacs_sketch_seed6304.json"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Evaluate using only 20 samples per domain"
    )

    args = parser.parse_args()

    evaluate_and_plot_controlled_study(
        args.study_type,
        args.data_root,
        args.ckpt_dir,
        args.num_workers,
        save_dir=args.save_dir,
        split_json=args.split_json,
        debug=args.debug
    )