import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import ToTensor

class FCNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(FCNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x1 = self.leaky_relu(self.fc1(x))
        x2 = self.leaky_relu(self.fc2(x1))
        x3 = self.leaky_relu(self.fc3(x2))
        x4 = self.fc4(x3)
        return x4

def calculate_accuracy(model: FCNeuralNetwork, dataloader: DataLoader[MNIST], device: str) -> float:
    """计算模型在给定数据加载器上的准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    train_data: MNIST = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader: DataLoader[MNIST] = DataLoader(train_data, batch_size=1024)

    test_data: MNIST = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    test_dataloader: DataLoader[MNIST] = DataLoader(test_data, batch_size=256)

    model: FCNeuralNetwork = FCNeuralNetwork(784, 10).to(device)
    optimizer: Adam = optim.Adam(model.parameters(), lr=0.01)
    ce_loss: CrossEntropyLoss = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            losses = ce_loss(y_hat, y)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        test_accuracy: float = calculate_accuracy(model, test_dataloader, device)
        print(f"Epoch {epoch + 1}: Test Accuracy: {test_accuracy:.2f}%")
