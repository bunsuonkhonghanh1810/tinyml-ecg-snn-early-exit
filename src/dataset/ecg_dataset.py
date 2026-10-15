import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.x = torch.from_numpy(data["x"]).float().unsqueeze(1)
        self.y = torch.from_numpy(data["y"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def compute_class_weights(y_tensor, num_classes=5):
    counts = torch.bincount(y_tensor, minlength=num_classes).float()
    total = len(y_tensor)
    weights = total / (num_classes * torch.clamp(counts, min=1.0))
    normalized_weights = weights / weights.mean()
    return normalized_weights
