# Code Explanation of `data_loaders.py`
## 1. Importing libraries
```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

```
- torch — the main PyTorch library.
- Dataset — the base class for all PyTorch datasets.
- datasets — ready-made datasets (e.g., FashionMNIST, CIFAR10).
- ToTensor — converts images to a PyTorch tensor.
- matplotlib.pyplot — used for displaying images.
## 2. Loading FashionMNIST training data
