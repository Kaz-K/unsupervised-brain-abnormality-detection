import torch.nn as nn

from models.blocks import conv5x5
from models.blocks import ConvBlock
from models.blocks import ResidualBlock
from models.blocks import UpsampleBlock
from models.utils import apply_init_kaiming


class Decoder(nn.Module):
    initial_scale = 4

    def __init__(self, input_dim, z_dim, filters, activation, final_activation):
        super().__init__()

        self.conv1 = ConvBlock(z_dim, filters[0])

        layers = []
        for i in range(1, len(filters)):
            in_channels = filters[i-1]
            out_channels = filters[i]

            layers.append(nn.Sequential(
                UpsampleBlock(in_channels, out_channels, activation=activation),
                ResidualBlock(out_channels, out_channels, activation=activation),
            ))

        self.module_list = nn.ModuleList(layers)

        if final_activation == 'tanh':
            self.final = nn.Sequential(
                conv5x5(filters[-1], input_dim),
                nn.Tanh(),
            )
        elif final_activation == 'none':
            self.final = nn.Sequential(
                conv5x5(filters[-1], input_dim),
            )
        else:
            raise NotImplementedError

        apply_init_kaiming(self, activation)

    def forward(self, z):
        z = self.conv1(z)

        for i, f in enumerate(self.module_list):
            z = f(z)

        return self.final(z)
