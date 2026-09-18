import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.v2 as T

def run_cue_conflict_evaluation(
    heads_dict, 
    metadata_file="./task1/data/cue_conflicts/conflict_metadata.json", 
    image_dir="./task1/data/cue_conflicts",
    splits_file="./task1/data/stl10_splits.json"
):
    if not os.path.exists(metadata_file):
        print(f"Skipping cue conflict evaluation: {metadata_file} not found.")
        return {}

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        
    with open(splits_file, "r") as f:
        splits = json.load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Re-instantiate backbones to ensure clean evaluation
    from task1.models.backbones import ResNet50Wrapper, ViTB16Wrapper, OpenCLIPWrapper
    backbones = {
        "ResNet-50": ResNet50Wrapper().to(device),
        "ViT-B/16": ViTB16Wrapper().to(device),
        "CLIP-ViT-B/32": OpenCLIPWrapper(device=device),
    }
    
    clip_text_embeds = backbones["CLIP-ViT-B/32"].get_text_prompts(splits["classes"], device=device)
    
    base_tf = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToDtype(torch.float32, scale=True),
    ])
    norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    results = {}
    
    for name, backbone in backbones.items():
        head = heads_dict.get(name, None)
        models_to_run = [(f"{name}_head", head, False)]
        if name == "CLIP-ViT-B/32":
            models_to_run.append((f"{name}_zeroshot", None, True))
            
        for eval_name, clf_head, is_zero_shot in models_to_run:
            n_shape = 0
            n_texture = 0
            n_other = 0
            n_total = 0
            
            for filename, data in metadata.items():
                img_path = os.path.join(image_dir, filename)
                # Visual Rejection Rule Implementation:
                # If you deleted failed images from the folder manually, this skips them.
                if not os.path.exists(img_path):
                    continue
                    
                img = Image.open(img_path).convert("RGB")
                x = base_tf(img).unsqueeze(0)
                x_norm = norm_transform(x).to(device)
                
                with torch.no_grad():
                    f = backbone(x_norm)
                    if is_zero_shot:
                        logits = (f @ clip_text_embeds.T) * 100.0
                    else:
                        logits = clf_head(f)
                        
                    pred = logits.argmax(dim=-1).item()
                    
                if pred == data["shape_idx"]:
                    n_shape += 1
                elif pred == data["texture_idx"]:
                    n_texture += 1
                else:
                    n_other += 1
                n_total += 1
                
            if n_total == 0:
                continue
                
            valid_decisions = n_shape + n_texture
            shape_bias = (n_shape / valid_decisions * 100.0) if valid_decisions > 0 else 0.0
            coverage = (valid_decisions / n_total * 100.0) if n_total > 0 else 0.0
            
            results[eval_name] = {
                "n_total_accepted": n_total,
                "n_shape": n_shape,
                "n_texture": n_texture,
                "n_other": n_other,
                "shape_bias_percent": shape_bias,
                "coverage_percent": coverage
            }
            
    return results