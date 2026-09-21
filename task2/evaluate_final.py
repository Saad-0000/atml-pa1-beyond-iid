import os
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from shared.pacs import PACSDataset, get_pacs_transforms, PACS_CLASSES
from shared.pacs_protocol import create_or_load_pacs_splits
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.methods.source_only import SourceOnlyModel
from task2.evaluation.metrics import evaluate_loader
from task2.evaluation.domain_separability import compute_domain_separability
from task2.evaluation.class_analysis import evaluate_per_class_and_confusions


def evaluate_all(data_root, checkpoint_dir, output_dir="./task2/results", num_workers=0,
                 split_json="splits/pacs_sketch_seed6304.json", debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = create_or_load_pacs_splits(data_root, split_json)
    _, eval_tf = get_pacs_transforms()

    source_val_loaders = {}
    for d in ["photo", "art_painting", "cartoon"]:
        ds_val = PACSDataset(splits["sources"][d]["val"], transform=eval_tf)

        if debug:
            ds_val = Subset(ds_val, range(min(20, len(ds_val))))

        source_val_loaders[d] = DataLoader(
            ds_val,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers
        )

    target_ds = PACSDataset(splits["target"]["sketch"], transform=eval_tf)

    if debug:
        target_ds = Subset(target_ds, range(min(20, len(target_ds))))

    target_loader = DataLoader(
        target_ds,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers
    )

    methods = ["source_only", "dan", "dann", "cdan"]
    records = []
    class_summaries = {}

    src_only_acc = None

    for m in methods:
        ckpt_path = os.path.join(checkpoint_dir, f"{m}_val1.0_best.pt")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint for {m}: {ckpt_path}")

        backbone = ResNet18Backbone()
        classifier = ClassifierHead(512, 7)
        model = SourceOnlyModel(backbone, classifier).to(device)

        state_dict = torch.load(ckpt_path, map_location=device)
        inference_state = {
            k: v for k, v in state_dict.items()
            if k in model.state_dict()
        }

        model.load_state_dict(inference_state, strict=True)
        model.eval()

        # 1. Source Validation Domains
        src_metrics = {}

        for d, ldr in source_val_loaders.items():
            acc, f1 = evaluate_loader(model, ldr, device)
            src_metrics[f"{d}_acc"] = acc
            src_metrics[f"{d}_f1"] = f1

        mean_src_acc = np.mean([
            src_metrics[f"{d}_acc"]
            for d in source_val_loaders
        ])

        mean_src_f1 = np.mean([
            src_metrics[f"{d}_f1"]
            for d in source_val_loaders
        ])

        # 2. Target Evaluation
        tgt_acc, tgt_f1 = evaluate_loader(
            model, target_loader, device
        )

        if m == "source_only":
            src_only_acc = tgt_acc

        delta_acc = tgt_acc - src_only_acc

        # 3. Domain Separability
        separability = compute_domain_separability(
            model.backbone,
            source_val_loaders,
            target_loader,
            device
        )

        # 4. Per-Class and Confusions
        per_class, _ = evaluate_per_class_and_confusions(
            model,
            target_loader,
            device
        )

        class_summaries[m] = per_class

        row = {
            "Method": m.upper(),
            "Photo Acc": round(src_metrics["photo_acc"], 2),
            "Art Acc": round(src_metrics["art_painting_acc"], 2),
            "Cartoon Acc": round(src_metrics["cartoon_acc"], 2),
            "Mean Src Acc": round(mean_src_acc, 2),
            "Mean Src F1": round(mean_src_f1, 2),
            "Sketch Acc": round(tgt_acc, 2),
            "Sketch F1": round(tgt_f1, 2),
            "Sketch Delta": round(delta_acc, 2),
            "Separability (%)": round(separability, 2)
        }

        records.append(row)

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(records)

    csv_path = os.path.join(
        output_dir,
        "task2_main_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print("\n--- Task 2 Main Results ---")
    print(df.to_string(index=False))

    with open(
        os.path.join(output_dir, "per_class_breakdown.json"),
        "w"
    ) as f:
        json.dump(class_summaries, f, indent=2)

    print(
        f"\nSaved results to {csv_path} "
        "and per_class_breakdown.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "--output_dir",
        type=str,
        default="./task2/results"
    )

    parser.add_argument(
        "--split_json",
        type=str,
        default="splits/pacs_sketch_seed6304.json"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="0 for local testing, 2+ for Colab"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Evaluate using only 20 samples per domain"
    )

    args = parser.parse_args()

    evaluate_all(
        args.data_root,
        args.ckpt_dir,
        args.output_dir,
        args.num_workers,
        args.split_json,
        args.debug
    )