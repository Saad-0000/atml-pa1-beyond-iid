import os
import yaml
import copy
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, get_pacs_transforms
from shared.pacs_protocol import create_or_load_pacs_splits

from task2.models.backbone import (
    ResNet18Backbone,
    freeze_bn_running_stats,
)

from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator

from task2.methods.source_only import SourceOnlyModel
from task2.methods.dan import DANModel
from task2.methods.dann import DANNModel
from task2.methods.cdan import CDANModel

from task2.evaluation.metrics import evaluate_loader


DEFAULT_CONFIG = {
    "seed": 6304,
    "data_root": "./data/PACS",
    "save_dir": "./task2/results/checkpoints",
    "split_json": "splits/pacs_sketch_seed6304.json",
    "batch_size_per_source": 8,
    "batch_size_target": 24,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "max_epochs": 30,
    "patience": 5,
    "param_val": 1.0,
    "num_workers": 0,
}


def load_config(config_path):
    """
    Load a Task 2 YAML config and resolve its optional inherit entry.
    """

    if not config_path:
        return {}

    config_path = os.path.abspath(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parent = config.pop("inherit", None)

    if parent:
        inherited = load_config(
            os.path.join(
                os.path.dirname(config_path),
                parent,
            )
        )

        inherited.update(config)

        return inherited

    return config


def build_loaders(
    data_root,
    split_json,
    seed=6304,
    num_workers=0,
    batch_size_per_source=8,
    batch_size_target=24,
):
    splits = create_or_load_pacs_splits(
        data_root,
        split_json,
        seed=seed,
    )

    train_tf, eval_tf = get_pacs_transforms()

    source_train_loaders = {}
    source_val_loaders = {}

    for d in [
        "photo",
        "art_painting",
        "cartoon",
    ]:

        ds_tr = PACSDataset(
            splits["sources"][d]["train"],
            transform=train_tf,
        )

        ds_val = PACSDataset(
            splits["sources"][d]["val"],
            transform=eval_tf,
        )

        source_train_loaders[d] = DataLoader(
            ds_tr,
            batch_size=batch_size_per_source,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )

        source_val_loaders[d] = DataLoader(
            ds_val,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
        )

    ds_target = PACSDataset(
        splits["target"]["sketch"],
        transform=train_tf,
    )

    target_loader = DataLoader(
        ds_target,
        batch_size=batch_size_target,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    empty = [
        d
        for d, loader in source_train_loaders.items()
        if len(loader) == 0
    ]

    empty.extend(
        d
        for d, loader in source_val_loaders.items()
        if len(loader) == 0
    )

    if len(target_loader) == 0:
        empty.append("sketch target")

    if empty:
        raise ValueError(
            "Insufficient PACS training samples after splitting for: "
            f"{', '.join(empty)}. "
            "Reduce the batch size or provide a complete PACS dataset."
        )

    return (
        source_train_loaders,
        source_val_loaders,
        target_loader,
    )


def build_method_model(
    method_name,
    backbone,
    classifier,
    param_val,
):

    if method_name == "source_only":

        return SourceOnlyModel(
            backbone,
            classifier,
        )

    elif method_name == "dan":

        return DANModel(
            backbone,
            classifier,
            lambda_mmd=param_val,
        )

    elif method_name == "dann":

        discriminator = DomainDiscriminator(
            in_features=512
        )

        return DANNModel(
            backbone,
            classifier,
            discriminator,
            grl_max=param_val,
        )

    elif method_name == "cdan":

        discriminator = DomainDiscriminator(
            in_features=512 * 7
        )

        return CDANModel(
            backbone,
            classifier,
            discriminator,
            grl_max=param_val,
        )

    else:

        raise ValueError(
            f"Unknown method: {method_name}"
        )


def calculate_gradient_norm(model):
    """
    Calculate the global L2 norm of all gradients.
    """

    total_norm_squared = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            return float("inf")

        parameter_norm = parameter.grad.detach().norm(2)

        total_norm_squared += parameter_norm.item() ** 2

    return total_norm_squared ** 0.5


def train_task2(
    method_name,
    data_root,
    save_dir,
    param_val=1.0,
    seed=6304,
    num_workers=0,
    batch_size_per_source=8,
    batch_size_target=24,
    lr=1e-4,
    weight_decay=1e-4,
    max_epochs=30,
    patience=5,
    steps_per_epoch=None,
    split_json="splits/pacs_sketch_seed6304.json",
):

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    if max_epochs < 1:
        raise ValueError(
            "max_epochs must be at least 1"
        )

    if patience < 1:
        raise ValueError(
            "patience must be at least 1"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Method: {method_name}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"GRL max: {param_val}")

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    (
        source_tr_ldrs,
        source_val_ldrs,
        target_ldr,
    ) = build_loaders(
        data_root,
        split_json,
        seed=seed,
        num_workers=num_workers,
        batch_size_per_source=batch_size_per_source,
        batch_size_target=batch_size_target,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    backbone = ResNet18Backbone()

    classifier = ClassifierHead(
        in_features=512,
        num_classes=7,
    )

    model = build_method_model(
        method_name,
        backbone,
        classifier,
        param_val,
    ).to(device)

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    # ---------------------------------------------------------
    # Checkpoint tracking
    # ---------------------------------------------------------

    best_f1 = -1.0
    patience_counter = 0
    best_weights = None

    # ---------------------------------------------------------
    # Number of training steps
    # ---------------------------------------------------------

    if steps_per_epoch is None:
        steps_per_epoch = max(
            len(ldr)
            for ldr in source_tr_ldrs.values()
        )

    if steps_per_epoch < 1:
        raise ValueError(
            "steps_per_epoch must be at least 1"
        )

    total_steps = (
        max_epochs * steps_per_epoch
    )

    curr_step = 0

    # ---------------------------------------------------------
    # Iterators
    # ---------------------------------------------------------

    iter_sources = {
        d: iter(ldr)
        for d, ldr in source_tr_ldrs.items()
    }

    iter_target = iter(target_ldr)

    history = {
        "train_loss": [],
        "val_macro_f1": [],
    }

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(max_epochs):

        model.train()

        # Keep BatchNorm running statistics frozen.
        freeze_bn_running_stats(model)

        epoch_loss = 0.0

        epoch_cls_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_alpha = 0.0

        for step in range(steps_per_epoch):

            # -------------------------------------------------
            # Fetch source batches
            # -------------------------------------------------

            x_s_list = []
            y_s_list = []

            for d in [
                "photo",
                "art_painting",
                "cartoon",
            ]:

                try:

                    imgs, lbls, _ = next(
                        iter_sources[d]
                    )

                except StopIteration:

                    iter_sources[d] = iter(
                        source_tr_ldrs[d]
                    )

                    imgs, lbls, _ = next(
                        iter_sources[d]
                    )

                x_s_list.append(imgs)
                y_s_list.append(lbls)

            x_s = torch.cat(
                x_s_list,
                dim=0,
            ).to(device)

            y_s = torch.cat(
                y_s_list,
                dim=0,
            ).to(device)

            # -------------------------------------------------
            # Fetch target batch
            # -------------------------------------------------

            try:

                x_t, _, _ = next(
                    iter_target
                )

            except StopIteration:

                iter_target = iter(
                    target_ldr
                )

                x_t, _, _ = next(
                    iter_target
                )

            x_t = x_t.to(device)

            # -------------------------------------------------
            # Training progress
            # -------------------------------------------------

            progress = (
                float(curr_step)
                / float(total_steps)
            )

            # -------------------------------------------------
            # Forward + loss
            # -------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, logs = model.train_step(
                x_s=x_s,
                y_s=y_s,
                x_t=x_t,
                progress=progress,
            )

            # -------------------------------------------------
            # Backward
            # -------------------------------------------------

            loss.backward()

            # -------------------------------------------------
            # Gradient diagnostics
            # -------------------------------------------------

            grad_norm_before_clip = (
                calculate_gradient_norm(model)
            )

            if not np.isfinite(
                grad_norm_before_clip
            ):
                raise RuntimeError(
                    "Non-finite gradient detected "
                    f"at epoch {epoch + 1}, "
                    f"step {step + 1}."
                )

            # -------------------------------------------------
            # Gradient clipping
            # -------------------------------------------------

            clipped_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            # -------------------------------------------------
            # Optimizer update
            # -------------------------------------------------

            optimizer.step()

            # -------------------------------------------------
            # Keep BN statistics frozen
            # -------------------------------------------------

            freeze_bn_running_stats(model)

            # -------------------------------------------------
            # Track diagnostics
            # -------------------------------------------------

            epoch_loss += logs["total_loss"]

            epoch_cls_loss += logs.get(
                "cls_loss",
                0.0,
            )

            epoch_domain_loss += logs.get(
                "domain_loss",
                0.0,
            )

            epoch_alpha += logs.get(
                "alpha",
                0.0,
            )

            epoch_grad_norm += (
                grad_norm_before_clip
            )

            # -------------------------------------------------
            # Print first step of every epoch
            # -------------------------------------------------


            if step == 0:
                print(
                    f"Step {step+1:03d} | "
                    f"Cls: {logs.get('cls_loss', 0):.4f} | "
                    f"Domain: {logs.get('domain_loss', 0):.4f} | "
                    f"Total: {logs['total_loss']:.4f} | "
                    f"Alpha: {logs.get('alpha', 0):.4f} | "
                    f"Grad: {grad_norm_before_clip:.2f} | "
                    f"FeatS: {logs.get('feat_s_norm', 0):.2f} | "
                    f"FeatT: {logs.get('feat_t_norm', 0):.2f} | "
                    f"Logit: {logs.get('domain_logit_mean', 0):.2f}"
                )

            curr_step += 1

        # -----------------------------------------------------
        # Epoch averages
        # -----------------------------------------------------

        avg_loss = (
            epoch_loss / steps_per_epoch
        )

        avg_cls_loss = (
            epoch_cls_loss
            / steps_per_epoch
        )

        avg_domain_loss = (
            epoch_domain_loss
            / steps_per_epoch
        )

        avg_alpha = (
            epoch_alpha
            / steps_per_epoch
        )

        avg_grad_norm = (
            epoch_grad_norm
            / steps_per_epoch
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        f1_list = []

        for d, v_ldr in source_val_ldrs.items():

            _, f1 = evaluate_loader(
                model,
                v_ldr,
                device,
            )

            f1_list.append(f1)

        mean_val_f1 = float(
            np.mean(f1_list)
        )

        history["train_loss"].append(
            avg_loss
        )

        history["val_macro_f1"].append(
            mean_val_f1
        )

        # -----------------------------------------------------
        # Epoch output
        # -----------------------------------------------------

        print(
            f"Epoch {epoch+1:02d}/{max_epochs:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Cls: {avg_cls_loss:.4f} | "
            f"Domain: {avg_domain_loss:.4f} | "
            f"Alpha: {avg_alpha:.4f} | "
            f"Grad: {avg_grad_norm:.2f} | "
            f"F1: {mean_val_f1:.2f}%"
        )

        # -----------------------------------------------------
        # Checkpoint selection
        # -----------------------------------------------------

        if mean_val_f1 > best_f1:

            best_f1 = mean_val_f1

            patience_counter = 0

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            print(
                f"  New best checkpoint: "
                f"{best_f1:.2f}%"
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement. "
                f"Patience: "
                f"{patience_counter}/{patience}"
            )

            if patience_counter >= patience:

                print(
                    "Early stopping triggered after "
                    f"{patience} epochs without improvement."
                )

                break

    # ---------------------------------------------------------
    # Save best model
    # ---------------------------------------------------------

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    save_path = os.path.join(
        save_dir,
        f"{method_name}_val{param_val}_best.pt",
    )

    if best_weights is None:
        raise RuntimeError(
            "No best model was saved."
        )

    torch.save(
        best_weights,
        save_path,
    )

    print()
    print(
        f"Model saved to {save_path}"
    )

    print(
        f"Best Source Val Macro-F1: "
        f"{best_f1:.2f}%"
    )

    return save_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help=(
            "YAML config, optionally "
            "inheriting from base.yaml"
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "source_only",
            "dan",
            "dann",
            "cdan",
        ],
    )

    parser.add_argument(
        "--data_root",
        type=str,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
    )

    parser.add_argument(
        "--split_json",
        type=str,
    )

    parser.add_argument(
        "--param_val",
        type=float,
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        help=(
            "0 for local testing, "
            "2+ for Colab"
        ),
    )

    parser.add_argument(
        "--batch_size_per_source",
        type=int,
    )

    parser.add_argument(
        "--batch_size_target",
        type=int,
    )

    parser.add_argument(
        "--lr",
        type=float,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
    )

    parser.add_argument(
        "--patience",
        type=int,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Run 1 epoch with 2 steps "
            "for pipeline verification"
        ),
    )

    cli_values = vars(
        parser.parse_args()
    )

    config = DEFAULT_CONFIG.copy()

    config.update(
        load_config(
            cli_values.pop("config")
        )
    )

    debug = cli_values.pop("debug")

    for key, value in cli_values.items():

        if value is not None:
            config[key] = value

    if "method" in config:

        config["method_name"] = config.pop(
            "method"
        )

    if "method_name" not in config:

        parser.error(
            "--method is required unless "
            "it is supplied by --config"
        )

    if debug:

        config.update(
            max_epochs=1,
            steps_per_epoch=2,
        )

        print(
            "DEBUG MODE: "
            "Running 1 epoch, 2 steps."
        )

    train_task2(
        **config
    )