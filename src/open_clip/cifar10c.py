import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CIFAR10C(Dataset):
    
    def __init__(self, root, corruption='gaussian_noise', severity=None, transform=None):
        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        data_path = os.path.join(root, f'{corruption}.npy')
        label_path = os.path.join(root, 'labels.npy')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Corruption file not found: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Labels file not found: {label_path}")
        
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        
        if severity is not None:
            if severity < 1 or severity > 5:
                raise ValueError("Severity must be between 1 and 5")

            start_idx = (severity - 1) * 10000
            end_idx = severity * 10000
            self.data = self.data[start_idx:end_idx]
            self.labels = self.labels[start_idx:end_idx]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


def get_cifar10c_dataloader(root, corruption='gaussian_noise', severity=None, 
                            transform=None, batch_size=128, num_workers=4):
    dataset = CIFAR10C(
        root=root,
        corruption=corruption,
        severity=severity,
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset