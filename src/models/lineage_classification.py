from asyncio.log import logger
import comet_ml 
import pandas as pd 
import numpy as np 
import torch
import os 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning.loggers import CometLogger
from PIL import Image
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import pathlib 
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data'))
from dataset import CellDataset

class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.lr = 1e-4
        self.momentum = 1e-2
        self.weight_decay = 1e-6
        self.accuracy = Accuracy()

        self.stack = nn.Sequential(
            # up
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            # more up!
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

            # down
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            # linear and classification layers (dont need softmax, computed in gradient calculation)
            nn.Flatten(),
            nn.Linear(128*704,1024),
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
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log("train_accuracy", acc, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log("val_accuracy", acc, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr, 
            momentum=self.momentum, 
            weight_decay=self.weight_decay,
        )
        return optimizer