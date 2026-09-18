import argparse
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from task1.configs import configs
import torchvision.datasets as datasets


def create_splits(
    data_root: str,
    output_dir: str,
    smoke_test: bool = False,
    seed: int = configs.SEED,
):
    os.makedirs(output_dir, exist_ok=True)
    train_set = datasets.STL10(root=data_root, split="train", download=True)
    test_set = datasets.STL10(root=data_root, split="test", download=True)

    y_train = np.array(train_set.labels)
    y_test = np.array(test_set.labels)

    # 1. Stratified 80/20 split on official train partition
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=configs.TRAIN_VAL_SPLIT_RATIO, random_state=seed
    )
    train_idx, val_idx = next(sss.split(np.zeros(len(y_train)), y_train))

    # 2. Balanced test subset
    test_subset_size = (
        configs.SMOKE_TEST_SAMPLES if smoke_test else configs.TEST_SUBSET_SIZE
    )
    test_ratio = test_subset_size / len(y_test)
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    _, eval_idx = next(sss_test.split(np.zeros(len(y_test)), y_test))

    splits_data = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "eval_test_indices": eval_idx.tolist(),
        "classes": train_set.classes,
    }

    out_path = os.path.join(output_dir, "stl10_splits.json")
    with open(out_path, "w") as f:
        json.dump(splits_data, f, indent=2)

    print(f"Splits saved to {out_path}")
    print(
        f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test Evaluation Subset: {len(eval_idx)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=configs.DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default="./task1/data")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=configs.SMOKE_TEST,
        help="Use miniature subset for testing on laptop",
    )
    args = parser.parse_args()
    create_splits(args.data_root, args.output_dir, args.smoke_test)