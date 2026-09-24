import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_dataloaders(batch_size, method='vanilla'):
    torch.manual_seed(6304)
    np.random.seed(6304)

    # CIFAR-10 Transforms
    transform_train_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if method == 'gcsc':
        transform_train_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    transform_train_list.append(transforms.ToTensor())
    
    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose([transforms.ToTensor()])

    cifar10_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_val_full = datasets.CIFAR10(root='./data', train=True, transform=transform_test)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 90/10 stratified split using seed 6304
    targets = cifar10_full.targets
    train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.1, stratify=targets, random_state=6304)
    
    train_loader = DataLoader(Subset(cifar10_full, train_idx), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(cifar10_val_full, val_idx), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def get_cifar100_unknowns(batch_size):
    """Isolates the fixed near and far semantic unknowns."""
    transform_test = transforms.Compose([transforms.ToTensor()])
    cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    near_classes = ['bus', 'pickup_truck', 'motorcycle', 'tractor', 'wolf', 'fox', 'leopard', 'camel']
    far_classes = ['bottle', 'bowl', 'chair', 'clock', 'keyboard', 'mushroom', 'sunflower', 'wardrobe']
    
    class_to_idx = cifar100_test.class_to_idx
    near_idx = [class_to_idx[c] for c in near_classes]
    far_idx = [class_to_idx[c] for c in far_classes]

    near_indices = [i for i, target in enumerate(cifar100_test.targets) if target in near_idx]
    far_indices = [i for i, target in enumerate(cifar100_test.targets) if target in far_idx]

    near_loader = DataLoader(Subset(cifar100_test, near_indices), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    far_loader = DataLoader(Subset(cifar100_test, far_indices), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return near_loader, far_loader