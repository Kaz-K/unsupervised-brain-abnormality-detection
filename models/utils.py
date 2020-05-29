import torch.nn as nn
from torch.nn.utils import spectral_norm


def get_act_func(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    else:
        raise NotImplementedError


def apply_spectral_norm(net):
    def _add_spectral_norm(m):
        classname = m.__class__.__name__
        print(classname)
        if classname.find('Conv2d') != -1:
            m = spectral_norm(m)
        elif classname.find('Linear') != -1:
            m = spectral_norm(m)

    print('applying normalization [spectral_norm]')
    net.apply(_add_spectral_norm)


def apply_init_kaiming(net, nonlinearity):
    def _weight_init_kaiming(m):
        classname = m.__class__.__name__
        print(classname)
        if classname.find('Conv2d') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity=nonlinearity)
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity=nonlinearity)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialization method [kaiming_normal]')
    net.apply(_weight_init_kaiming)
