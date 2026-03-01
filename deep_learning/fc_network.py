import torch

"""
手写pytorch模块学习pytorch底层原理
"""

class LinearLayer:
    """
    全连接线性层
    """
    def __init__(self, input_dim, output_dim):
        self.weight = torch.randn(input_dim, output_dim, requires_grad=True)
        self.bias = torch.randn(1, output_dim, requires_grad=True)

    def forward(self, x):
        # x.shape = (#records, #features)
        # weights.shape = (#features, #neurons)
        # bias.shape = (1, #neurons)
        # output.shape = (#records, #neurons)
        return torch.matmul(x, self.weight) + self.bias

    def parameters(self):
        return [self.weight, self.bias] # 返回自己参数的引用，用于给optimizer用于更新参数


class SGDOptimizer:
    """
    优化器
    """
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        # 清空所有参数的grad
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad(): # 不记录这不操作的grad信息
            for param in self.params:
                param -= self.lr * param.grad # 根据grad信息更新参数


if __name__ == '__main__':
    linear = LinearLayer(input_dim=10, output_dim=5)
    optimizer = SGDOptimizer(linear.parameters(), lr=0.01)

    x = torch.randn(8, 10)
    for epoch in range(100):
        output = linear.forward(x)
        loss = torch.mean((output - torch.randn_like(output))**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")



"""
torch.nn.Module 是所有神经网络类的"超级基类", 继承这个类可以有以下超能力：
- 使用 nn.Parameter 自动注册参数后，可以使用 model.parameters() 一次性调出模型所有需要更新的参数的引用
- 使用 model.to(device) 一次性将模型所有参数迁移至GPU
- model.zero_grad() 一次性清空所有参数的grad
- torch.save(model.state_dict(), ...) 一次性保存模型参数
- 会自动管理所有子模块

nn.Module 强制用户把前向传播写 forward() 里面，调用 model(x) 的时候 nn.Module 会自动调用前向传播方法。
"""