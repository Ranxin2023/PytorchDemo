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
```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

```
- `FashionMNIST` — 70,000 grayscale clothing images (28×28).
- `root="data"` — folder where the dataset will be stored.
- `train=True` — load the 60,000 training images.
- `download=True` — download it automatically if not present.
- transform=ToTensor() — converts each image:
## 4. Label names for pretty printing
```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

```
- FashionMNIST stores labels as 0–9 integers.
- This dictionary maps each label to a human-readable name.
## 5. Displaying 9 random samples
```python
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

```
- Step-by-step:
1. **Create a new figure** of size 8×8.
2. **Prepare a grid** of 3 rows × 3 columns = 9 images.
3. For each grid cell:
    - Pick a random index using `torch.randint`.
    - Get the image + label.
    - Add a subplot.
    - Set title to the label ("T-Shirt", etc.).
    - Remove axis ticks.
    - Show image (`img.squeeze()` removes the channel dimension).
## 7. Custom dataset implementation
```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

```
- Purpose:
    - This allows you to load **any folder of images** using a CSV annotation file.
- Important concepts:
    - `annotations_file`: CSV file containing `filename,label`
    - `pd.read_csv` loads the annotation table.
    - `img_dir` is where the images actually live.
    - `transform` applies transforms to images (resize, normalize, etc).
    - `target_transform` applies transforms to labels.
