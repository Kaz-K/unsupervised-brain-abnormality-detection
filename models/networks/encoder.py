import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import conv1x1
from models.blocks import conv5x5
from models.blocks import ConvBlock
from models.blocks import ResidualBlock
from models.utils import apply_init_kaiming


class Encoder(nn.Module):

    def __init__(self, input_dim, z_dim, filters, activation):
        super().__init__()

        layers = []
        for i, filter in enumerate(filters):
            if i == 0:
                in_channels = input_dim
                out_channels = filter
                layers.append(nn.Sequential(
                    conv5x5(in_channels, out_channels),
                    nn.AvgPool2d(2),
                ))

            else:
                in_channels = filters[i-1]
                out_channels = filter
                layers.append(nn.Sequential(
                    ResidualBlock(in_channels, out_channels, activation=activation),
                    nn.AvgPool2d(2),
                ))

        self.module_list = nn.ModuleList(layers)

        self.conv1_1 = ConvBlock(filters[-1], z_dim, activation=activation)
        self.conv2_1 = conv1x1(z_dim, z_dim)
        self.conv2_2 = conv1x1(z_dim, z_dim)

        apply_init_kaiming(self, activation)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, with_var=False):
        for i, f in enumerate(self.module_list):
            x = f(x)

        x = self.conv1_1(x)
        mu = self.conv2_1(x)
        logvar = self.conv2_2(x)

        if self.training:
            z = self._reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            if with_var:
                return mu, logvar
            else:
                return mu

    def get_feature_maps(self, x):
        features = []
        for i, f in enumerate(self.module_list):
            x = f(x)
            h = F.interpolate(x, scale_factor=2*(2**i))
            features.append(h)

        return features
