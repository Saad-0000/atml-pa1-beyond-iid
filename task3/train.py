import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, ConcatDataset

from shared.pacs import PACSDataset, get_pacs_transforms
from shared.pacs_protocol import create_or_load_pacs_splits

from task3.models.backbone import FrozenBNResNet18, ClassifierHead
from task3.methods.dan_dg import compute_dan_dg_loss
from task3.methods.sam import SAM

class PACSDGDataset(torch.utils.data.Dataset):
    def __init__(self, ds, domain_idx):
        self.ds = ds
        self.domain_idx = domain_idx
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, label, _ = self.ds[idx]
        return img, label, self.domain_idx

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to PACS dataset')
    parser.add_argument('--debug', action='store_true', help='Run short pipeline for local testing')
    return parser.parse_args()

def main():
    args = get_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.get('seed', 6304)) 

    method = config['method']
    batch_size = config['batch_size'] 
    max_epochs = 1 if args.debug else config['max_epochs'] 
    
    train_tf, _ = get_pacs_transforms()
    splits = create_or_load_pacs_splits(args.data_root)

    source_domains = ["photo", "art_painting", "cartoon"]
    train_datasets = []
    
    for i, d in enumerate(source_domains):
        ds = PACSDataset(splits["sources"][d]["train"], transform=train_tf)
        train_datasets.append(PACSDGDataset(ds, i))
        
    dataset = ConcatDataset(train_datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 

    backbone = FrozenBNResNet18().to(device)
    classifier = ClassifierHead().to(device)
    criterion = nn.CrossEntropyLoss()

    params = list(backbone.parameters()) + list(classifier.parameters())
    
    if method == 'sam':
        optimizer = SAM(
            params, 
            torch.optim.AdamW, 
            rho=config.get('rho', 0.05), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = torch.optim.AdamW(
            params, 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )

    for epoch in range(max_epochs):
        backbone.train()
        classifier.train()
        
        for batch_idx, (images, labels, domains) in enumerate(loader):
            images, labels, domains = images.to(device), labels.to(device), domains.to(device)

            if method == 'sam':
                features = backbone(images)
                loss = criterion(classifier(features), labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                criterion(classifier(backbone(images)), labels).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                features = backbone(images)
                cls_loss = criterion(classifier(features), labels)
                
                if method == 'dan_dg':
                    lambda_dg = config.get('lambda_dg', 1.0)
                    mmd_loss = compute_dan_dg_loss(features, domains, lambda_dg=lambda_dg)
                    loss = cls_loss + mmd_loss
                else:
                    loss = cls_loss 
                    
                loss.backward()
                optimizer.step()
                
            if args.debug and batch_idx >= 1: 
                print("Debug mode: stopping epoch early.")
                break 

        print(f"Epoch {epoch+1}/{max_epochs} Complete. Method: {method}")

    os.makedirs('task3/results', exist_ok=True)
    torch.save({
        'backbone': backbone.state_dict(),
        'classifier': classifier.state_dict()
    }, f'task3/results/{method}_checkpoint.pth')
    print(f"Saved {method} to task3/results/")

if __name__ == '__main__':
    main()