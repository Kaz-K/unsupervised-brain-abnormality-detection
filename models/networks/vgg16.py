from collections import namedtuple

import torch
import torch.nn.functional as F
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):

        super(Vgg16, self).__init__()
        pretrained = models.vgg16(pretrained=False)
        pretrained.load_state_dict(torch.load('./saved_models/vgg16-397923af.pth'))
        vgg_pretrained_features = pretrained.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.norm1_2 = torch.nn.InstanceNorm2d(64)
        self.norm2_2 = torch.nn.InstanceNorm2d(128)
        self.norm3_3 = torch.nn.InstanceNorm2d(256)
        self.norm4_3 = torch.nn.InstanceNorm2d(512)

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, normalize_fm, layers=['h_relu1_2', 'h_relu2_2', 'h_relu3_3', 'h_relu4_3']):
        x = x.repeat((1, 3, 1, 1))
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        h_relu2_2 = F.interpolate(h_relu2_2, scale_factor=2)
        h_relu3_3 = F.interpolate(h_relu3_3, scale_factor=4)
        h_relu4_3 = F.interpolate(h_relu4_3, scale_factor=8)

        if normalize_fm:
            h_relu1_2 = self.norm1_2(h_relu1_2)
            h_relu2_2 = self.norm2_2(h_relu2_2)
            h_relu3_3 = self.norm3_3(h_relu3_3)
            h_relu4_3 = self.norm4_3(h_relu4_3)

        output_layers = []
        for layer in layers:
            if layer == 'h_relu1_2':
                output_layers.append(h_relu1_2)
            elif layer == 'h_relu2_2':
                output_layers.append(h_relu2_2)
            elif layer == 'h_relu3_3':
                output_layers.append(h_relu3_3)
            elif layer == 'h_relu4_3':
                output_layers.append(h_relu4_3)

        return output_layers
