import pandas as pd 
import numpy as np 
import os 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CellDataset(Dataset):
    def __init__(self, images_path, label_path):
        self.images_path = images_path
        self.labels = pd.read_csv(label_path)
        self.tensor = transforms.ToTensor()
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        img_path, label = self.labels.iloc[idx]['filename'], self.labels.iloc[idx]['class']
        img = Image.open(os.path.join(self.images_path, img_path))
        return self.tensor(img), label
