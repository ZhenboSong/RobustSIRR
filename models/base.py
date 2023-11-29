import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from copy import deepcopy


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, conv_layer, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            if isinstance(v, str) and v[0]=='P': # CFR support
                v=int(v[1:])
                conv2d = ProbConv2d(conv_layer(in_channels, v, kernel_size=3, padding=1))
            else:
                conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=not isinstance(conv2d, ProbConv2d))]
            else:
                layers += [conv2d, nn.ReLU(inplace=not isinstance(conv2d, ProbConv2d))]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = conv_layer(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = conv_layer(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def discriminator(k=4, fc=True):
    return Discriminator(models.resnet18(pretrained=True, progress=False), k, fc=fc)

class Discriminator(nn.Module):
    def __init__(self, base_net, k=4, fc=True):
        super().__init__()
        self.base_net=base_net
        if hasattr(self.base_net,'fc'):
            self.base_net.fc=nn.Linear(base_net.fc.in_features, k) if fc else nn.Sequential()
        else:
            self.base_net.classifier=nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, k),
            )
        self.fc=fc

    def forward(self, x):
        x = self.base_net(x)
        if self.fc:
            # 对某一维度的行进行softmax运算
            x = F.softmax(x, dim=-1)
        return x

class ProbConv2d(nn.Module):
    def __init__(self, conv_layer, eps=1e-8):
        super().__init__()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        if hasattr(conv_layer,'K'):
            self.K=conv_layer.K

        self.conv_mean=conv_layer
        self.conv_std=deepcopy(conv_layer)
        self.eps=eps

    def forward(self, x, softmax_attention=None):
        if softmax_attention is None:
            x_mean=self.conv_mean(x)
            x_log_var=self.conv_std(x)
        else:
            x_mean = self.conv_mean(x, softmax_attention)
            x_log_var = self.conv_std(x, softmax_attention)
        #x_log_var = torch.where(torch.isinf(x_log_var.exp()), torch.full_like(x_log_var, 0), x_log_var)
        x_log_var = x_log_var.clip(max=10)
        return self.reparameterize(x_mean, x_log_var)

    # 随机生成隐含向量
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z, mu, logvar