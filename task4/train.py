import argparse
import yaml
import torch
import torch.nn as nn
import os
from data.make_splits import get_dataloaders
from models.resnet_cifar import CIFARResNet18
from methods.proser import train_proser_epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _ = get_dataloaders(cfg['batch_size'], cfg['method'])
    
    num_classes = 15 if cfg['method'] == 'proser' else 10
    model = CIFARResNet18(num_classes=num_classes).to(device)
    
    if cfg['method'] == 'proser':
        # Init from Vanilla[cite: 2]
        ckpt = torch.load('task4/results/vanilla_best.pth')
        
        # Extract the 10-class fc weights so they don't cause a shape mismatch
        fc_w = ckpt.pop('fc.weight')
        fc_b = ckpt.pop('fc.bias')
        
        # Load all the backbone (feature extractor) weights
        model.load_state_dict(ckpt, strict=False)
        
        # Manually copy the 10 known class weights into the new 15-class layer
        with torch.no_grad():
            model.fc.weight[:10] = fc_w
            model.fc.bias[:10] = fc_b
            nn.init.xavier_uniform_(model.fc.weight[10:]) # Random init dummy classifiers[cite: 2]
            nn.init.zeros_(model.fc.bias[10:])
        
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg['learning_rate'], 
        momentum=cfg['momentum'], 
        weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    os.makedirs('task4/results', exist_ok=True)
    
    for epoch in range(cfg['epochs']):
        epoch_loss = 0.0
        batches = 0
        
        if cfg['method'] == 'proser':
            # train_proser_epoch doesn't currently return loss, so we'll just track epoch number
            train_proser_epoch(model, train_loader, optimizer, criterion, device)
        else:
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
                
        scheduler.step()
        
        # Validate solely on CIFAR-10 known[cite: 2]
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                preds = logits[:, :10].argmax(dim=1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)
                
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"task4/results/{cfg['method']}_best.pth")
            
        # Print progress
        if cfg['method'] == 'proser':
            print(f"Epoch {epoch+1:03d}/{cfg['epochs']} | Val Acc: {val_acc:.4f} (Best: {best_val_acc:.4f})")
        else:
            avg_loss = epoch_loss / max(1, batches)
            print(f"Epoch {epoch+1:03d}/{cfg['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} (Best: {best_val_acc:.4f})")
            
    print(f"Finished {cfg['method']}. Best Val Acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()