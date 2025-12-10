# src/dataset.py
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

def get_transforms(image_size=224, mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

class TumorDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        root_dir: path to data/processed
        split: 'train' | 'val' | 'test'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        split_dir = self.root_dir / split
        classes = [p.name for p in split_dir.iterdir() if p.is_dir()]
        classes = sorted(classes)
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            cdir = split_dir / c
            for p in cdir.iterdir():
                if p.suffix.lower() in (".jpg",".jpeg",".png"):
                    self.samples.append((p, self.class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"No images found in {split_dir}. Check your data path and run split_data.py first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def make_loader(root_dir, split="train", batch_size=32, image_size=224, num_workers=0):
    transform = get_transforms(image_size=image_size, mode=("train" if split=="train" else "eval"))
    dataset = TumorDataset(root_dir=root_dir, split=split, transform=transform)
    shuffle = True if split=="train" else False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    return loader
