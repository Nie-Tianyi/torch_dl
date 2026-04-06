from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernal_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.swish(out)

        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.swish(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels=64, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y