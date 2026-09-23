import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import torch.nn as nn
import copy
import csv
from sklearn.metrics import f1_score
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
    parser.add_argument('--lambda_dg', type=float, default=None, help='Override lambda_dg from config')
    parser.add_argument('--rho', type=float, default=None, help='Override rho from config')
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
    
    # Overrides
    if args.lambda_dg is not None: config['lambda_dg'] = args.lambda_dg
    if args.rho is not None: config['rho'] = args.rho
    
    train_tf, test_tf = get_pacs_transforms()
    splits = create_or_load_pacs_splits(args.data_root)

    source_domains = ["photo", "art_painting", "cartoon"]
    train_datasets = []
    val_datasets = []
    
    for i, d in enumerate(source_domains):
        ds_train = PACSDataset(splits["sources"][d]["train"], transform=train_tf)
        train_datasets.append(PACSDGDataset(ds_train, i))
        
        ds_val = PACSDataset(splits["sources"][d]["val"], transform=test_tf)
        val_datasets.append(PACSDGDataset(ds_val, i))
        
    dataset = ConcatDataset(train_datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
    
    val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

    os.makedirs('task3/results', exist_ok=True)
    csv_file = open(f'task3/results/{method}_training_log.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Cls_Loss', 'MMD_Loss', 'Total_Loss', 'Train_Acc', 'Val_Macro_F1'])
    
    best_val_f1 = 0.0

    for epoch in range(max_epochs):
        backbone.train()
        classifier.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_mmd_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, domains) in enumerate(loader):
            images, labels, domains = images.to(device), labels.to(device), domains.to(device)

            if method == 'sam':
                features = backbone(images)
                logits = classifier(features)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                criterion(classifier(backbone(images)), labels).backward()
                optimizer.second_step(zero_grad=True)
                
                cls_loss = loss
                mmd_loss = torch.tensor(0.0)
            else:
                optimizer.zero_grad()
                features = backbone(images)
                logits = classifier(features)
                cls_loss = criterion(logits, labels)
                
                if method == 'dan_dg':
                    lambda_dg = config.get('lambda_dg', 1.0)
                    mmd_loss = compute_dan_dg_loss(features, domains, lambda_dg=lambda_dg)
                    loss = cls_loss + mmd_loss
                else:
                    loss = cls_loss 
                    mmd_loss = torch.tensor(0.0)
                    
                loss.backward()
                optimizer.step()
                
            # Track metrics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd_loss.item() if method == 'dan_dg' else 0.0
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
                
            if args.debug and batch_idx >= 1: 
                break 

        # Compute averages
        avg_loss = total_loss / len(loader)
        avg_cls_loss = total_cls_loss / len(loader)
        train_acc = 100. * correct / total
        
        # Validation
        backbone.eval()
        classifier.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                logits = classifier(backbone(images.to(device)))
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())
                if args.debug: break
                
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'backbone': backbone.state_dict(),
                'classifier': classifier.state_dict()
            }, f'task3/results/{method}_best_checkpoint.pth')
        
        avg_mmd_loss = (total_mmd_loss / len(loader)) if method == 'dan_dg' else 0.0
        csv_writer.writerow([epoch+1, avg_cls_loss, avg_mmd_loss, avg_loss, train_acc, val_f1])
        csv_file.flush()
        
        metrics_str = f"Epoch {epoch+1:02d}/{max_epochs} [{method}] | Acc: {train_acc:.2f}% | Cls Loss: {avg_cls_loss:.4f} | Val F1: {val_f1:.4f}"
        if method == 'dan_dg':
            metrics_str += f" | MMD Loss: {avg_mmd_loss:.4f} | Total Loss: {avg_loss:.4f}"
            
        print(metrics_str)

    csv_file.close()
    
    # Save final model as well
    torch.save({
        'backbone': backbone.state_dict(),
        'classifier': classifier.state_dict()
    }, f'task3/results/{method}_final_checkpoint.pth')
    print(f"Saved {method} best and final checkpoints to task3/results/")

if __name__ == '__main__':
    main()