import torch
from torch import nn, optim
from tqdm import tqdm


class SimpleCNN(nn.Module):
    def __init__(self, activation_function: nn.Module):
        super(SimpleCNN, self).__init__()
        self.act = activation_function()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # 必须加 BN，因为 x^2 对方差太敏感
            self.act,
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.act,
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.act,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(128, 64),
            self.act,
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_unit(x)
        x = x.view(x.size(0), -1)
        x = self.fc_unit(x)
        return x


def test_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100. * correct / total
    return test_acc



def train_config(activation_function, device, trainloader, testloader, epochs=20):
    model = SimpleCNN(activation_function=activation_function).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_losses = []
    train_accs = []
    test_accs = []

    print(f"\n开始训练模型（激活函数：{activation_function.__name__}）")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 计算本轮指标
        epoch_loss = running_loss / len(trainloader)
        epoch_train_acc = 100. * correct / total
        epoch_test_acc = test_model(model, testloader, device)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(epoch_test_acc)


        print(f"Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Test Acc: {epoch_test_acc:.2f}%")

    return train_losses, train_accs, test_accs
