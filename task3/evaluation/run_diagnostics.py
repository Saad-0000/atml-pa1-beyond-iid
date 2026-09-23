import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from task3.models.backbone import FrozenBNResNet18, ClassifierHead
from task3.evaluation.diagnostics import compute_source_separability, compute_sharpness_proxy
from shared.pacs import PACSDataset, get_pacs_transforms
from shared.pacs_protocol import create_or_load_pacs_splits
from task3.train import PACSDGDataset
import json

def run_diagnostics():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    backbone = FrozenBNResNet18().to(device)
    classifier = ClassifierHead().to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    classifier.load_state_dict(ckpt['classifier'])
    
    backbone.eval()
    classifier.eval()

    _, eval_tf = get_pacs_transforms()
    splits = create_or_load_pacs_splits(args.data_root)
    
    source_domains = ["photo", "art_painting", "cartoon"]
    val_datasets = []
    
    for i, d in enumerate(source_domains):
        ds_val = PACSDataset(splits["sources"][d]["val"], transform=eval_tf)
        val_datasets.append(PACSDGDataset(ds_val, i))
        
    val_dataset = ConcatDataset(val_datasets)
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Running diagnostics for {args.checkpoint}...")

    # Extract features for separability
    all_features = []
    all_domains = []
    
    with torch.no_grad():
        for images, _, domains in loader:
            images = images.to(device)
            feats = backbone(images)
            all_features.append(feats.cpu())
            all_domains.append(domains)
            
    all_features = torch.cat(all_features, dim=0)
    all_domains = torch.cat(all_domains, dim=0)
    
    print("Computing Source-Domain Separability...")
    sep_score = compute_source_separability(all_features, all_domains)
    print(f"Separability Score: {sep_score:.4f}")
    
    print("Computing Sharpness Proxy...")
    criterion = nn.CrossEntropyLoss()
    sharpness = compute_sharpness_proxy(backbone, classifier, loader, criterion, device)
    print(f"Sharpness Proxy (Cross-Entropy Increase): {sharpness:.4f}")

    # Save diagnostics
    results_file = "task3/results/diagnostics_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            try:
                results_data = json.load(f)
            except json.JSONDecodeError:
                results_data = {}
    else:
        results_data = {}
        
    method_name = os.path.basename(args.checkpoint).replace('_checkpoint.pth', '').replace('_best', '').replace('_final', '')
    if method_name not in results_data:
        results_data[method_name] = {}
        
    results_data[method_name]["Separability"] = float(sep_score)
    results_data[method_name]["Sharpness"] = float(sharpness)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Diagnostics saved to {results_file}")

if __name__ == '__main__':
    run_diagnostics()

