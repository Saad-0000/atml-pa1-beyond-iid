import torch
import torch.nn as nn
import numpy as np

def train_proser_epoch(model, dataloader, optimizer, criterion, device, beta=1.0, gamma=0.1):
    model.train()
    pre_mixup, post_mixup = model.get_proser_layers()
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Split mini-batch in half[cite: 2]
        half = images.size(0) // 2
        img_cls, lbl_cls = images[:half], labels[:half]
        img_mix, lbl_mix = images[half:], labels[half:]
        
        if len(img_cls) == 0: continue
        optimizer.zero_grad()
        
        # 1. Classifier Placeholders
        logits_cls = model(img_cls)
        # Push 2nd highest to dummy[cite: 2]
        loss_known = criterion(logits_cls[:, :10], lbl_cls)
        
        # 2. Data Placeholders (Manifold Mixup)
        with torch.no_grad():
            h_mix = pre_mixup(img_mix)
            
        # Shuffle for mixup ensuring y_i != y_j[cite: 2]
        idx = torch.randperm(h_mix.size(0))
        for i in range(len(idx)):
            if lbl_mix[i] == lbl_mix[idx[i]]:
                idx[i] = (idx[i] + 1) % len(idx)
                
        lam = np.random.beta(2.0, 2.0)
        h_tilde = lam * h_mix + (1 - lam) * h_mix[idx]
        
        feat_tilde = post_mixup(h_tilde)
        feat_tilde = torch.flatten(feat_tilde, 1)
        logits_tilde = model.fc(feat_tilde)
        
        # Train towards dummy classifiers[cite: 2]
        dummy_labels = torch.randint(10, 15, (logits_tilde.size(0),)).to(device)
        loss_data = criterion(logits_tilde, dummy_labels)
        
        loss = loss_known + (beta * loss_known) + (gamma * loss_data)
        loss.backward()
        optimizer.step()