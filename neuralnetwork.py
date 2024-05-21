import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class NN(nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NN(784, 10)
x = torch.randn(64, 784)
print(f"the shape of model is: {model(x).shape}")

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 1

# Load Data
train_dataset = datasets.MNIST(
    root="./dataset", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(datasets=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root="./dataset", train=True, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(datasets=train_dataset, batch_size=batch_size, shuffle=True)
