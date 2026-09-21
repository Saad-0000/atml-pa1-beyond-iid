import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]

class PACSDataset(Dataset):
    """
    Standard PACS Dataset reader. Expects directory structure:
    root_dir/
      domain/
        class_name/
          image.jpg
    """
    def __init__(self, samples, transform=None):
        self.samples = samples  # List of tuples: (image_path, label_idx, domain_name)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, domain

def get_pacs_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    eval_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform