import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from contextlib import redirect_stdout
def transfer_learning():
    # ✅ Pre-trained model
    model = models.resnet18(pretrained=True)

    # Replace final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)

    # ✅ Fine-tuning strategy: freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # ✅ Data loaders
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # ✅ Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # ✅ Training loop (1 epoch demo)
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break  # just 1 batch for demo
    print(f"Train loss: {loss.item():.4f}")

    # ✅ Evaluation (metrics + validation strategy)
    from sklearn.metrics import accuracy_score

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            break  # just 1 batch for demo

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation accuracy (1 batch demo): {acc:.4f}")

def transfer_learning_redirect_output():
    with open("./outputs/transfer_learning.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            transfer_learning()
