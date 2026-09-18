import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms.v2 as T

from task1.configs import configs
from task1.analysis.representation import compute_cosine_stability, plot_umap_projections
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


def extract_features(model, loader, transform_fn, device, norm_transform, smoke_test=False):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if smoke_test and batch_idx >= configs.MAX_VAL_BATCHES:
                break
            if transform_fn is not None:
                x = transform_fn(x)
            x_norm = norm_transform(x).to(device)
            f = model(x_norm)
            features.append(f.cpu())
            labels.append(y)
            
    return torch.cat(features, dim=0), torch.cat(labels, dim=0).numpy()


def run_feature_analysis(splits_file=configs.SPLITS_FILE, data_dir=configs.DATA_ROOT, output_dir=configs.OUTPUT_DIR, smoke_test=configs.SMOKE_TEST):
    os.makedirs(output_dir, exist_ok=True)
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
    
    transformations = {
        "grayscale": apply_grayscale,
        "hue_rotation": apply_hue_rotation,
        "translation_32px": lambda img: apply_cardinal_translation(img, delta=32, direction="right"),
        "patch_shuffle": lambda img: apply_patch_shuffle(img, grid_size=4, seed=configs.SEED),
    }
    
    stability_results = {}
    
    for name, backbone in backbones.items():
        print(f"Computing feature stability metrics for: {name}")
        stability_results[name] = {}
        
        # Clean baseline features
        clean_feats, labels = extract_features(
            backbone, eval_loader, None, device, norm_transform, smoke_test=smoke_test
        )
        
        for t_name, t_fn in transformations.items():
            trans_feats, _ = extract_features(
                backbone, eval_loader, t_fn, device, norm_transform, smoke_test=smoke_test
            )
            
            # Compute I_T cosine stability
            i_t = compute_cosine_stability(clean_feats, trans_feats)
            stability_results[name][f"cosine_stability_{t_name}"] = i_t

            # Sanitize model name for file paths (replace / with -)
            safe_name = name.replace("/", "-")
            
            # Plot and save combined UMAP projection
            plot_path = os.path.join(output_dir, f"umap_{safe_name}_{t_name}.png")
            plot_umap_projections(
                clean_feats=clean_feats.numpy(),
                trans_feats=trans_feats.numpy(),
                labels=labels,
                title=f"{name}: Clean vs {t_name} (I_T = {i_t:.3f})",
                save_path=plot_path,
                seed=configs.SEED
            )
            
    out_file = os.path.join(output_dir, "feature_stability_results.json")
    with open(out_file, "w") as f:
        json.dump(stability_results, f, indent=2)
        
    print(f"Representation stability metrics saved to {out_file}")
    return stability_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_file", type=str, default=configs.SPLITS_FILE)
    parser.add_argument("--data_dir", type=str, default=configs.DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default=configs.OUTPUT_DIR)
    parser.add_argument("--smoke-test", action="store_true", default=configs.SMOKE_TEST)
    args = parser.parse_args()
    
    run_feature_analysis(args.splits_file, args.data_dir, args.output_dir, args.smoke_test)