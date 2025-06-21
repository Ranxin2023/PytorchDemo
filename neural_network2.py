import torch
import torch.nn as nn
import torch.nn.functional as F

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # MNIST: 28x28 input
        self.fc2 = nn.Linear(128, 10)     # 10 output classes

    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

