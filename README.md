# Pytorch Demo
## Table of Contents
- [Concepts](#concepts)
    - [Pytorch](#1pytorch)
    - [Tensor](#2-tensor)
- [Packages]()
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
#### 💡 What is Transfer Learning?
- Transfer learning is a machine learning technique where:
    - We reuse a model that was already trained on one task
    - And adapt it to a new but related task

#### 🧠 Core idea
- Instead of training a model from scratch (which can take lots of data + time):

    - You take a model trained on a large dataset (e.g., ImageNet with millions of images).
    - You keep most of the model’s learned features (filters, weights).
    - You modify (or fine-tune) the last few layers to match your new task (e.g., classifying cats/dogs, or CIFAR-10).

#### 🌟 Why does it work?
- Early layers in deep networks learn **generic features** (e.g., edges, textures, shapes)
- Later layers learn **task-specific features**:
    - Keep early layers (they detect general patterns)
    - Change/adjust later layers for your specific problem

#### 📌 Example
- Imagine:
    - A ResNet-18 trained on ImageNet (1000 categories)
    - You want to classify 10 types of flowers
    -  Instead of starting from random weights:
        - Load the ResNet-18 pretrained model
        - Replace the final classifier layer to output 10 classes
        - Fine-tune this new model on your flower dataset
    - This saves time and often achieves better accuracy, especially when your dataset is small!

#### Feature extraction
| Approach               | What happens                                                 |
| ---------------------- | ------------------------------------------------------------ |
| **Feature extraction** | Freeze all base layers; only train final classifier          |
| **Fine-tuning**        | Unfreeze some/all base layers; continue training with low LR |

#### Difference between normal learning and transfer learning
- In transfer learning:
    - You **start with a model that’s already been trained** on a large dataset (e.g. ImageNet)
    - You reuse **the learned features** (weights) from that model
    - You reuse the learned features (weights) from that model
### 6. ResNet-18?
#### 💡 What is ResNet-18?
- **ResNet-18** is a deep **convolutional neural network (CNN)** architecture that was introduced by Microsoft Research in the 2015 paper:
- `"Deep Residual Learning for Image Recognition"` by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
- The **“18”** refers to **the number of layers with learnable weights** in the network (18 layers = 17 convolutional + 1 fully connected layer).
#### 

### 7. Loss Function
####  What is a loss function?
- A **loss function** is a general term.
    -  It measures **how far off** your model’s prediction is from the true target.
- The purpose of any loss function:
    - Quantify the error
    - Guide the model’s learning (via backpropagation + gradient descent)
- Example loss functions:
    - `CrossEntropyLoss` (for classification)
    - `MSELoss` (Mean Squared Error, for regression)
    - `L1Loss` (Mean Absolute Error)
- What is CrossEntropyLoss?
    - `CrossEntropyLoss` is one specific type of loss function.
    -  It is designed for:
        - Multi-class classification tasks
        - It combines LogSoftmax + Negative Log-Likelihood Loss (NLLLoss) in one

### 8. FashionMNIST
#### What is FashionMNIST?
- FashionMNIST is a dataset of **grayscale clothing images**, designed to be a drop-in replacement for the original MNIST digit dataset.
- It contains:
    - 60,000 training images
    - 10,000 test images
    - Image size: 28 × 28
    - 1 channel (grayscale)
- Each image shows a clothing item, such as a shoe or a shirt.
#### Why FashionMNIST was created
- 
### 9. CIFAR-10 Classes
#### Overview
- CIFAR-10 is a dataset of **real-world color images**, originally collected by the Canadian Institute for Advanced Research (CIFAR).
- It contains:
    - 50,000 training images
    - 10,000 test images
    - Image size: 32 × 32
    - 3 channels (RGB color)
- These are small but diverse natural images.
- CIFAR-10 Classes
|**Label**| **Class**|**Example**|
| ------- | -------- | --------- |
| 0       | Airplane   | ✈️      |
| 1       | Automobile | 🚗      |
| 2       | Bird       | 🐦      |
| 3       | Cat        | 🐱      |
| 4       | Deer       | 🦌      |
| 5       | Dog        | 🐕      |
| 6       | Frog       | 🐸      |
| 7       | Horse      | 🐎      |
| 8       | Ship       | 🚢      |
| 9       | Truck      | 🚛      |

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
- Original ResNet-18 ends with model.fc → output 1000 classes (ImageNet).
- We replace it with a new layer:

    - Input = same as original `model.fc` input

    - Output = 10 (CIFAR-10 classes)


#### 📌 Freeze early layers
```python 
for param in model.parameters():
    param.requires_grad = False

```
- Freeze all layers → prevent updates during training.
- Useful when we want to only train the classifier (final layer).
```python
for param in model.fc.parameters():
    param.requires_grad = True

```
#### 📌 Data transforms
```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

```
- Define preprocessing:

    - Resize CIFAR-10 (32×32) images to 224×224 (ResNet-18 expects 224×224)

    - Convert PIL Image → PyTorch Tensor


#### 📌 Load CIFAR-10 dataset
```python
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

- Load CIFAR-10 training data
- Apply the transform
- DataLoader batches = 32, shuffles the data

```python
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32)
```
- Load CIFAR-10 validation/test data

#### 📌 Define loss + optimizer
```python
criterion = nn.CrossEntropyLoss()
```
CrossEntropyLoss combines:
- Softmax + negative log-likelihood
- Perfect for multi-class classification

## Packages
### 1. NumPy (<2)
#### What it is
- NumPy is the fundamental scientific computing library for Python.
#### Why you need it
- PyTorch internally interacts with NumPy a lot, especially when converting between:
    - PyTorch tensors ↔ NumPy arrays
    - Dataset preprocessing
    - Visualization and debugging
### Summary Table
| Package         | Purpose                      | Why Needed                                        |
| --------------- | ---------------------------- | ------------------------------------------------- |
| **numpy<2**     | Math arrays, vector ops      | Required by PyTorch + matplotlib                  |
| **torch**       | Deep learning engine         | Core of dataset, tensors, DataLoader              |
| **torchvision** | Vision datasets & transforms | Required for FashionMNIST, ToTensor, decode_image |
| **torchaudio**  | Audio tools                  | Not used here but version-matched; optional       |
| **matplotlib**  | Plotting images              | Required for dataset visualization                |
