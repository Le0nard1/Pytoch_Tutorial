import torch 
from torch import nn
import pytorch_lightning as pl 
from torchvision.datasets import MNIST
from torch.optim import Adam

train_loader = MNIST('', download = False)
encoder = nn.Linear(28*28, 10)

optimizer = Adam(encoder.parameters())

trainer = Trainer(max_epochs =10)
trainer.fit(encoder, train_loader, optimizer)

class Trainer:
    def __init__(max_epochs=10):
        self.max_epochs = max_epochs

    def fit(self, model, train_loader):
        for epoch in range(self.max_epochs):
            for batch in train_loader:
                x, y = batch
                x = x.view(x.size(0, -1))
                logit = model(x)
                loss = cross_entropy(logit, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


