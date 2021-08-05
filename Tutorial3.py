import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import pdb
import pytorch_lightning as pl

"""
PyTorch Lightning

1. model
2. optimizer
3. data
4. training loop
5. validation loop
"""

class ResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear( 28 * 28 , 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,10)
        self.do = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 +h1)
        logits = self.l3(do)
        return logits

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)
        logits = self(x)
        J = self.loss(logits, y)
        return {'loss':J}
    
    def train_dataloader(self):
        train_data = datasets.MNIST(
        "data", train=True, download=False, transform=transforms.ToTensor()
        )
        #train, val = random_split(train_data, [55000, 5000])
        train_loader = DataLoader(train_data, batch_size=32)
        #val_loader = DataLoader(val, batch_size=32)
        return train_loader
    
    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results

class ImageClassifier(nn.Module):
    def __init__(self):
        self.resnet = ResNet()

model = ResNet()


trainer = pl.Trainer(progress_bar_refresh_rate = 20, max_epochs=5)
trainer.fit(model)