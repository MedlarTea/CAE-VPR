from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from ..utils.serialization import load_checkpoint, copy_state_dict


__all__ = ['alexnet']

class Imagenet_matconvnet_alex(nn.Module):

    def __init__(self):
        super().__init__()
        self.meta = {'mean': [122.73954010009766, 114.89665222167969, 101.59194946289062],
                     'std': [1, 1, 1],
                     'imageSize': [227, 227, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[11, 11], stride=(4, 4))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(1, 1), padding=(2, 2), groups=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), groups=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), groups=2)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.fc6 = nn.Conv2d(256, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, input):
        x1 = self.conv1(input)
        x3 = self.relu1(x1)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x7 = self.relu2(x5)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x11 = self.relu3(x9)
        x12 = self.conv4(x11)
        x14 = self.relu4(x12)
        x15 = self.conv5(x14)
        # x17 = self.relu5(x15)
        # x18 = self.pool5(x17)
        # x19 = self.fc6(x18)
        # x21_preflatten = self.relu6(x19)
        # x21 = x21_preflatten.view(x21_preflatten.size(0), -1)
        # x22 = self.fc7(x21)
        # x24 = self.relu7(x22)
        # prediction = self.fc8(x24)
        # return prediction
        return x15

class ALEXNET(nn.Module):
    __cut_layers = { # alext
        'conv5':11,
        'conv4':9,
        'conv3':7,
        'conv2':4,
        'conv1':1,
    }

    def __init__(self, cut_layer="conv5", matconvnet=None):
        super(ALEXNET, self).__init__()
        self.cut_layer = cut_layer
        self.matconvnet = matconvnet

        alexnet = Imagenet_matconvnet_alex()
        # layers = list(alexnet.children())[:ALEXNET.__cut_layers[self.cut_layer]]
        self.base = alexnet
        
        self._init_params()

        layers = list(self.base.children())
        # for l in layers:
        #     for p in l.parameters():
        #         p.requires_grad = False

    def _init_params(self):
        # optional load pretrained weights from matconvnet
        if (self.matconvnet is not None):
            self.base.load_state_dict(torch.load(self.matconvnet))
            self.pretrained = True

    def forward(self, x):
        N,C,H,W = x.size()

        x = self.base(x)
        print(x.max())
        x = x.view(N, -1)

        return x

def alexnet(**kwargs):
    return ALEXNET(**kwargs)
