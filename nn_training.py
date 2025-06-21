import torch
import torch.nn as nn
from contextlib import redirect_stdout
from neural_network2 import Net2
from CNNDemo import CNN, training_cnn
def nn_traning(model):
    from torch.optim import SGD

    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Example dummy data
    inputs = torch.rand(64, 1, 28, 28)
    labels = torch.randint(0, 10, (64,))

    for epoch in range(1000):  # 100 epochs
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def redirect_nn_training():
    with open("./outputs/nn2.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("training neural net 2......")
            nn_traning(Net2())
    with open("./outputs/cnn_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("training CNN......")
            training_cnn()