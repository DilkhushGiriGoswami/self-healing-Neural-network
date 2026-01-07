import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train_model():
    transform = transforms.ToTensor()
    # train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform)

    loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    model = BaseModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "model.pth")
    return model
