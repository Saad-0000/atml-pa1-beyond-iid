import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.v2 as T
from torchvision.datasets import STL10
from tqdm import tqdm

from task1.configs import configs   


class VGGFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Use VGG19 to extract style and content features
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Standard Gatys style transfer layers
        self.content_layers = {'21': 'conv4_2'}
        self.style_layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        self.model = vgg

    def forward(self, x):
        features = {'content': {}, 'style': {}}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.content_layers:
                features['content'][self.content_layers[name]] = x
            if name in self.style_layers:
                features['style'][self.style_layers[name]] = x
        return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t()) / (d * h * w)
    return gram


def run_style_transfer(vgg, content_img, style_img, device, num_steps=200):
    """Optimizes a stylized image from a content (shape) and style (texture) target."""
    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.05)
    vgg_norm = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    content_features = vgg(vgg_norm(content_img.to(device)))['content']
    style_features = vgg(vgg_norm(style_img.to(device)))['style']
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    style_weight = 1e4
    content_weight = 1

    for step in range(num_steps):
        target_features = vgg(vgg_norm(target))
        
        # Content Loss
        content_loss = F.mse_loss(target_features['content']['conv4_2'], content_features['conv4_2'])
        
        # Style Loss
        style_loss = 0
        for layer in target_features['style']:
            target_gram = gram_matrix(target_features['style'][layer])
            style_gram = style_grams[layer]
            style_loss += F.mse_loss(target_gram, style_gram)

        loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep image in valid float range [0, 1]
        with torch.no_grad():
            target.clamp_(0, 1)

    return target.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=configs.DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default="./task1/data/cue_conflicts")
    parser.add_argument("--smoke-test", action="store_true", default=configs.SMOKE_TEST)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Initializing VGG19 on {device}...")
    vgg = VGGFeatures(device)

    # 1. Load Data
    base_tf = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToDtype(torch.float32, scale=True),
    ])
    dataset = STL10(root=args.data_dir, split="test", download=False, transform=base_tf)
    
    classes = dataset.classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Required class pairs for conflict generation
    pairs = [
        ("car", "bird"),
        ("truck", "horse"),
        ("ship", "cat"),
        ("horse", "airplane"),
        ("ship", "dog")
    ]

    # Curated source images selected from STL-10 contact sheets.
    curated_indices = {
        "airplane": [2904, 6407, 7145, 5267, 248, 4993, 1643, 6420],
        "bird": [2673, 6172, 4111, 4007, 6821, 5771, 119, 140],
        "car": [1827, 5028, 224, 1056, 7905, 2159, 4217, 7632],
        "cat": [509, 2775, 4818, 2961, 4241, 2890, 3546, 2857],
        "deer": [5469, 4454, 7690, 6828, 6036, 4273, 4199],
        "dog": [568, 451, 7631, 7536, 138, 2791, 5366],
        "horse": [6203, 6124, 2003, 7042, 2254, 7075, 7211],
        "ship": [1614, 1093, 3691, 4374, 4699, 7323, 3250],
        "truck": [7047, 5182, 4930, 2535, 7131, 1332, 7383, 4127],
    }

    required_classes = {class_name for pair in pairs for class_name in pair}
    missing_classes = required_classes - curated_indices.keys()
    if missing_classes:
        raise ValueError(
            f"Missing curated source images for: {sorted(missing_classes)}"
        )

    # Load only the selected source images and verify their labels.
    images_by_class = {}
    for class_name, indices in curated_indices.items():
        images_by_class[class_name] = []
        for index in indices:
            img, label = dataset[index]
            if classes[label] != class_name:
                raise ValueError(
                    f"Dataset index {index} is labelled {classes[label]}, "
                    f"not {class_name}."
                )
            images_by_class[class_name].append(img)

    total_images = 2 if args.smoke_test else 200
    images_per_pair = max(1, total_images // len(pairs))
    opt_steps = 200 if args.smoke_test else 400

    metadata = {}
    image_counter = 0

    print(f"Generating {total_images} cue-conflict images...")
    
    for shape_cls, texture_cls in pairs:
        for _ in range(images_per_pair):
            if image_counter >= total_images:
                break
                
            # Randomly decide direction (A->B or B->A) to keep it balanced
            if random.random() > 0.5:
                s_cls, t_cls = shape_cls, texture_cls
            else:
                s_cls, t_cls = texture_cls, shape_cls
                
            # Sample random images
            shape_img = random.choice(images_by_class[s_cls]).unsqueeze(0)
            texture_img = random.choice(images_by_class[t_cls]).unsqueeze(0)
            
            # Generate style transfer
            stylized = run_style_transfer(vgg, shape_img, texture_img, device, num_steps=opt_steps)
            
            # Convert back to PIL to save
            stylized_pil = T.ToPILImage()(stylized.squeeze(0).cpu())
            
            filename = f"conflict_{image_counter:03d}_{s_cls}-shape_{t_cls}-texture.jpg"
            filepath = os.path.join(args.output_dir, filename)
            stylized_pil.save(filepath)
            
            metadata[filename] = {
                "shape_class": s_cls,
                "texture_class": t_cls,
                "shape_idx": class_to_idx[s_cls],
                "texture_idx": class_to_idx[t_cls]
            }
            image_counter += 1
            print(f"Saved: {filename}")

    # Save metadata JSON needed for evaluation
    meta_path = os.path.join(args.output_dir, "conflict_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Generation complete. Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
