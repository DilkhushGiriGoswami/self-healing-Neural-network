from train_model import train_model
from monitor import monitor_performance
from self_heal import self_heal
import torch
from torchvision import datasets, transforms

model = train_model()

# test_data = datasets.MNIST('.', train=False, transform=transforms.ToTensor())
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

accuracy = monitor_performance(model, test_loader)
print("Initial Accuracy:", accuracy)

if accuracy < 0.85:
    model = self_heal(model)
    accuracy = monitor_performance(model, test_loader)
    print("Recovered Accuracy:", accuracy)
