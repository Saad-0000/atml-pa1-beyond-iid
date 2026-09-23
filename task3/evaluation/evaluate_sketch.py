import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from task3.models.backbone import FrozenBNResNet18, ClassifierHead
from sklearn.metrics import accuracy_score, f1_score
from shared.pacs import PACSDataset, get_pacs_transforms
from shared.pacs_protocol import create_or_load_pacs_splits

def evaluate_target():
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
    
    # Load Sketch Domain ONLY here
    dataset = PACSDataset(splits["target"]["sketch"], transform=eval_tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            logits = classifier(backbone(images.to(device)))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Sketch Target Results for {args.checkpoint}:")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

if __name__ == '__main__':
    evaluate_target()