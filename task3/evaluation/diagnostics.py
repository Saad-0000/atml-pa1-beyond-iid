import torch
import torch.nn as nn
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_source_separability(features, domain_labels):
    """
    Trains multinomial logistic regression (C=1) to predict domain (Photo, Art, Cartoon)[cite: 1].
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features.cpu().numpy(), domain_labels.cpu().numpy(), 
        test_size=0.30, random_state=6304 # 70/30 split using seed 6304[cite: 1]
    )
    clf = LogisticRegression(C=1.0, multi_class='multinomial', max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    separability_score = accuracy_score(y_test, preds)
    return separability_score

@torch.no_grad()
def compute_sharpness_proxy(backbone, classifier, dataloader, criterion, device, epsilon=0.05):
    """
    Measures cross-entropy increase after normalized gradient-ascent perturbation[cite: 1].
    """
    backbone.eval()
    classifier.eval()
    
    # Grab one fixed validation batch (32 per source = 96 total)
    batch = next(iter(dataloader))
    images, labels = batch[0].to(device), batch[1].to(device)

    # Enable gradients momentarily to find ascent direction
    with torch.enable_grad():
        features = backbone(images)
        logits = classifier(features)
        loss = criterion(logits, labels)
        
        grads = torch.autograd.grad(loss, list(backbone.parameters()) + list(classifier.parameters()))
        grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    
    # Apply perturbation[cite: 1]
    delta = [epsilon * g / (grad_norm + 1e-12) for g in grads]
    
    # Measure perturbed loss
    perturbed_backbone = copy.deepcopy(backbone)
    perturbed_classifier = copy.deepcopy(classifier)
    
    params = list(perturbed_backbone.parameters()) + list(perturbed_classifier.parameters())
    for p, d in zip(params, delta):
        p.data.add_(d)
        
    perturbed_logits = perturbed_classifier(perturbed_backbone(images))
    perturbed_loss = criterion(perturbed_logits, labels)
    
    delta_sharp = (perturbed_loss - loss) / loss
    return delta_sharp.item()