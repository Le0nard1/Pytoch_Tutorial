import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


# Define My Model
model = nn.Sequential(
    nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)
)

# Define my Optimizer
optimiser = optim.SGD(model.parameters(), lr=1e-2)

# Define my Loss
loss = nn.CrossEntropyLoss()

# Train Data
train_data = datasets.MNIST(
    "data", train=True, download=False, transform=transforms.ToTensor()
)
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)


# Training Loop
nb_epochs = 5

for epoch in range(nb_epochs):
    losses = list()
    for batch in train_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1. forward
        l = model(x)  # l: logits

        # 2. compute objective function
        J = loss(l, y)

        # 3. cleaning the gradient
        model.zero_grad()

        # 4. accumulate the partial derivatives of J with respect to param
        J.backward()

        # 5. step in the opposite direction of the gradient
        optimiser.step()
        # with toch.no_grad(): param = params - eta*params.grad
        losses.append(J.item())

    print(f"Epoch {epoch+1}, train loss: {torch.tensor(losses).mean():.2f}")
