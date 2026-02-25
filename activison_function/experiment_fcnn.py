import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def experiment_on_fcnn(activation_function):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 定义转换组合
    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(
        root="./data", train=True, transform=my_transforms, download=True
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        activation_function(),
        nn.Linear(128, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ce_loss = nn.CrossEntropyLoss()

    loss_hist = []
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        losses = ce_loss(y_hat, y)
        loss_hist.append(losses.item())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # 画出loss_hist
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist)
    plt.title(f"Training Loss with {activation_function.__name__}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    return loss_hist