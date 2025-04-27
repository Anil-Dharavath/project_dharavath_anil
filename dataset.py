# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from config import resize_x, resize_y

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        self.data = pd.read_csv(csv_file)
        
        # Map 3 (happy) -> smiling (1), others -> not smiling (0)
        self.data['emotion'] = self.data['emotion'].apply(lambda x: 1 if x == 3 else 0)
        
        # Train/Validation Split
        split = int(0.8 * len(self.data))
        if train:
            self.data = self.data.iloc[:split]
        else:
            self.data = self.data.iloc[split:]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = np.fromstring(self.data.iloc[idx]['pixels'], sep=' ', dtype=np.float32).reshape(resize_x, resize_y)
        img = np.expand_dims(img, 0)  # Add channel dimension
        
        label = int(self.data.iloc[idx]['emotion'])
        
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label

def unicornLoader(csv_file, batch_size, train=True):
    dataset = FER2013Dataset(csv_file, train=train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
