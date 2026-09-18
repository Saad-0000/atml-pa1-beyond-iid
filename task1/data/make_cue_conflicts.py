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
    gram = torch.mm(tensor, tensor.t())
    return gram


def run_style_transfer(vgg, content_img, style_img, device, num_steps=200):
    """Optimizes a stylized image from a content (shape) and style (texture) target."""
    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.05)

    content_features = vgg(content_img.to(device))['content']
    style_features = vgg(style_img.to(device))['style']
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    style_weight = 1e6
    content_weight = 1

    for step in range(num_steps):
        target_features = vgg(target)
        
        # Content Loss
        content_loss = F.mse_loss(target_features['content']['conv4_2'], content_features['conv4_2'])
        
        # Style Loss
        style_loss = 0
        for layer in target_features['style']:
            target_gram = gram_matrix(target_features['style'][layer])
            style_gram = style_grams[layer]
            b, c, h, w = target_features['style'][layer].shape
            style_loss += F.mse_loss(target_gram, style_gram) / (c * h * w)

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
        ("airplane", "bird"),
        ("car", "dog"),
        ("ship", "truck"),
        ("horse", "cat"),
        ("deer", "car")
    ]

    # Organize images by class for sampling
    images_by_class = {c: [] for c in classes}
    for i in range(len(dataset)):
        img, label = dataset[i]
        images_by_class[classes[label]].append(img)
        # To speed up indexing, only sample the first 50 per class
        if len(images_by_class[classes[label]]) >= 50:
            continue

    total_images = 2 if args.smoke_test else 200
    images_per_pair = max(1, total_images // len(pairs))
    opt_steps = 10 if args.smoke_test else 150

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