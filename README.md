# Pytorch Demo
## Concepts
### 1.PyTorch
- PyTorch is an open-source deep learning framework that provides:
    - **Tensors** (like NumPy arrays but with GPU acceleration)

    - **Autograd** (automatic differentiation for gradients)

    - **Modules** and layers to build neural networks

    - **Optimizers** to train models
    It’s widely used for research and production.

### 2. Tensor
-  A multi-dimensional array (like a matrix).
- In PyTorch:
```python
x = torch.tensor([[1, 2], [3, 4]])
```
- Tensors can be processed on CPU or GPU.
- They are the basic data structure for inputs, outputs, weights.

### 3. Neural Network (Fully Connected / Dense)
- A sequence of layers where:

    - Every neuron connects to every neuron in the next layer.

    - Example from `neural_network2.py`:
    ```python
    self.fc1 = nn.Linear(28*28, 128)  # input: 784 pixels, output: 128 neurons
    self.fc2 = nn.Linear(128, 10)     # output: 10 classes
    ```
### 4. Convolutional Neural Network (CNN)
- A neural network designed for image data.
- In `CNNDemo.py`:

```python
self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
self.pool = nn.MaxPool2d(2, 2)
self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
```
- CNNs use:

    - **Convolution layers** to extract features (edges, shapes)

    - **Pooling layers** to reduce size + computation

    - **Fully connected layers** at the end for classification.

### 5. Transfer Learning

### 6. ResNet-18?
#### 💡 What is ResNet-18?
- **ResNet-18** is a deep **convolutional neural network (CNN)** architecture that was introduced by Microsoft Research in the 2015 paper:
- `"Deep Residual Learning for Image Recognition"` by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
- The **“18”** refers to **the number of layers with learnable weights** in the network (18 layers = 17 convolutional + 1 fully connected layer).
#### 
## Coding Explanation
### 1. Neural Network
Full Code:
```python
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

```
Detailed explanation:
`class Net2(nn.Module):`
This defines your model `Net2`, which inherits from `nn.Module` — the base class for all neural networks in PyTorch.

```python
self.fc1 = nn.Linear(28*28, 128)

```
➡ fc1 = fully connected layer (dense layer)
➡ Takes input size 28*28 = 784 (MNIST image flattened)
➡ Outputs 128 features

```python
self.fc2 = nn.Linear(128, 10)
```

### 3. Transfer Learning

#### 📌 Import libraries：
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

```

➡ Import:

- PyTorch core + neural network (nn) + optimizer (optim)

- torchvision for datasets (CIFAR-10), transforms (preprocessing), models (pre-trained models)

- DataLoader to batch and shuffle data

#### 📌 Load pre-trained model
```python
model = models.resnet18(pretrained=True)
```
- Load a ResNet-18 model with weights trained on ImageNet.
- This gives a strong starting point (transfer learning).

#### 📌 Replace final layer
```python
model = models.resnet18(pretrained=True)

```

#### 
