from torch import nn

from models.utils import get_act_func


def conv5x5(in_channels, out_channels, stride=1, padding=2, bias=True):
    return nn.Conv2d(in_channels, out_channels, 5, stride, padding, bias=bias)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=bias)


def conv1x1(in_channels, out_channels, stride=1, padding=0, bias=True):
    return nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias=bias)


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2, activation='relu'):
        super().__init__()

        act_func = get_act_func(activation)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            act_func,
        )

    def forward(self, x):
        return self.upsample(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act_func = get_act_func(activation)

    def forward(self, x):
        h = self.act_func(self.bn1(self.conv1(x)))
        h = self.act_func(self.bn2(self.conv2(h)))
        return h


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3x3(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = lambda x: x

        self.act_func = get_act_func(activation)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(x)
        out = self.act_func(out)

        return out
