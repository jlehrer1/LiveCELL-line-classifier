import pandas as pd 
import numpy as np 
import torch
import os 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pathlib 

class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(520*704*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,7)
        )

    def forward(self, x):
        return self.stack(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y, weight=self.weights)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        return val_loss

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


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()

    dataset = CellDataset(
        images_path=os.path.join(here, '..', '..', 'images'),
        label_path=os.path.join(here, '..', '..', 'labels.csv'),
    )
    
    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    traindata = DataLoader(train, batch_size=8, num_workers=100)
    valdata = DataLoader(test, batch_size=8, num_workers=100)

    trainer = pl.Trainer(
        gpus=4, 
        auto_lr_find=True,
        max_epochs=100000, 
    )

    model = Net()
    trainer.fit(model, traindata, valdata)

