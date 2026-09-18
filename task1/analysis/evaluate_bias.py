import argparse
import json
from json.tool import main
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms.v2 as T

from task1.configs import configs
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


def evaluate_model_on_loader(
    model, head, loader, transform_fn, device, norm_transform, is_clip_zero_shot=False, text_embeds=None, smoke_test=False
):
    preds, confidences, targets, features = [], [], [], []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if smoke_test and batch_idx >= configs.MAX_VAL_BATCHES:
                break
            
            # Apply intervention on raw RGB floats before normalization
            if transform_fn is not None:
                x = transform_fn(x)
                
            x_norm = norm_transform(x).to(device)
            f = model(x_norm)
            features.append(f.cpu())
            
            if is_clip_zero_shot:
                # Scaled cosine similarity between image and text embeddings
                logits = (f @ text_embeds.T) * 100.0
            else:
                logits = head(f)
                
            probs = F.softmax(logits, dim=-1)
            conf, p = probs.max(dim=-1)
            
            preds.extend(p.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
            targets.extend(y.numpy())
            
    features = torch.cat(features, dim=0)
    acc = accuracy_score(targets, preds) if len(targets) > 0 else 0.0
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0) if len(targets) > 0 else 0.0
    mean_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
    
    return {
        "preds": np.array(preds),
        "targets": np.array(targets),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "mean_confidence": mean_conf,
        "features": features,
    }


def run_bias_evaluation(heads_dict, splits_file=configs.SPLITS_FILE, data_dir=configs.DATA_ROOT, smoke_test=configs.SMOKE_TEST):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(splits_file, "r") as f:
        splits = json.load(f)
        
    base_tf = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToDtype(torch.float32, scale=True),
    ])
    
    eval_dataset = Subset(
        STL10(root=data_dir, split="test", download=False, transform=base_tf),
        splits["eval_test_indices"]
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16 if smoke_test else configs.BATCH_SIZE,
        shuffle=False
    )
    
    norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    backbones = {
        "ResNet-50": ResNet50Wrapper().to(device),
        "ViT-B/16": ViTB16Wrapper().to(device),
        "CLIP-ViT-B/32": OpenCLIPWrapper(device=device),
    }
    
    clip_text_embeds = backbones["CLIP-ViT-B/32"].get_text_prompts(splits["classes"], device=device)
    
    evaluation_records = {}
    
    for name, backbone in backbones.items():
        print(f"Evaluating bias metrics for: {name}")
        head = heads_dict.get(name, None)
        
        models_to_run = [(f"{name}_head", head, False)]
        if name == "CLIP-ViT-B/32":
            models_to_run.append((f"{name}_zeroshot", None, True))
            
        for eval_name, clf_head, is_zero_shot in models_to_run:
            # 1. Clean Baseline
            clean = evaluate_model_on_loader(
                backbone, clf_head, eval_loader, None, device, norm_transform,
                is_clip_zero_shot=is_zero_shot, text_embeds=clip_text_embeds, smoke_test=smoke_test
            )
            
            # 2. Color: Grayscale
            gray = evaluate_model_on_loader(
                backbone, clf_head, eval_loader, apply_grayscale, device, norm_transform,
                is_clip_zero_shot=is_zero_shot, text_embeds=clip_text_embeds, smoke_test=smoke_test
            )
            
            # 2. Color: 180-degree Hue Rotation
            hue = evaluate_model_on_loader(
                backbone, clf_head, eval_loader, apply_hue_rotation, device, norm_transform,
                is_clip_zero_shot=is_zero_shot, text_embeds=clip_text_embeds, smoke_test=smoke_test
            )
            
            # 3. Patch Shuffle (4x4)
            patch = evaluate_model_on_loader(
                backbone, clf_head, eval_loader,
                lambda img: apply_patch_shuffle(img, grid_size=4, seed=configs.SEED),
                device, norm_transform,
                is_clip_zero_shot=is_zero_shot, text_embeds=clip_text_embeds, smoke_test=smoke_test
            )
            
            # 4. Translation curve: deltas 0, 8, 16, 32 across cardinal directions
            translation_results = {}
            for delta in [0, 8, 16, 32]:
                dir_accuracies = []
                dir_consistencies = []
                for direction in ["up", "down", "left", "right"]:
                    trans_out = evaluate_model_on_loader(
                        backbone, clf_head, eval_loader,
                        lambda img: apply_cardinal_translation(img, delta=delta, direction=direction),
                        device, norm_transform,
                        is_clip_zero_shot=is_zero_shot, text_embeds=clip_text_embeds, smoke_test=smoke_test
                    )
                    dir_accuracies.append(trans_out["accuracy"])
                    dir_consistencies.append(np.mean(clean["preds"] == trans_out["preds"]))
                    
                translation_results[f"delta_{delta}"] = {
                    "mean_accuracy": float(np.mean(dir_accuracies)),
                    "mean_consistency": float(np.mean(dir_consistencies)),
                }
                
            evaluation_records[eval_name] = {
                "clean": {
                    "accuracy": clean["accuracy"],
                    "macro_f1": clean["macro_f1"],
                    "mean_confidence": clean["mean_confidence"],
                },
                "grayscale": {
                    "accuracy_drop": clean["accuracy"] - gray["accuracy"],
                    "prediction_consistency": float(np.mean(clean["preds"] == gray["preds"])),
                },
                "hue_rotation": {
                    "accuracy_drop": clean["accuracy"] - hue["accuracy"],
                    "prediction_consistency": float(np.mean(clean["preds"] == hue["preds"])),
                },
                "patch_shuffle": {
                    "accuracy_drop": clean["accuracy"] - patch["accuracy"],
                    "prediction_consistency": float(np.mean(clean["preds"] == patch["preds"])),
                },
                "translation_sweep": translation_results,
            }
            
    return evaluation_records

if __name__ == "__main__":
    main()