from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from ..utils.serialization import load_checkpoint, copy_state_dict


__all__ = ['VGG', 'vgg16']


class VGG(nn.Module):
    __factory = {
        16: torchvision.models.vgg16,
    }

    __fix_layers = { # vgg16
        'conv6':28,
        'conv5':24,
        'conv4':17,
        'conv3':10,
        'conv2':5,
        'full':0
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, maxpooling=False, avgpooling=False,
                    train_layers='conv5', matconvnet=None, return_layer=None, isflatten=False):
        super(VGG, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.train_layers = train_layers
        self.feature_dim = 512
        self.matconvnet = matconvnet
        self.return_layer = return_layer
        self.avgpooling = avgpooling
        self.maxpooling = maxpooling
        self.isflatten = isflatten
        # Construct base (pretrained) resnet
        if depth not in VGG.__factory:
            raise KeyError("Unsupported depth:", depth) 
        vgg = VGG.__factory[depth](pretrained=pretrained)
        layers = list(vgg.features.children())[:-2]
        self.base = nn.Sequential(*layers) # capture only feature part and remove last relu and maxpool
        self.gap = None
        if maxpooling:
            self.gap = nn.AdaptiveMaxPool2d(1)  # maxpooling
        elif avgpooling:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self._init_params()

        if not pretrained:
            self.reset_params()
        else:
            layers = list(self.base.children())
            for l in layers[:VGG.__fix_layers[train_layers]]:
                for p in l.parameters():
                    p.requires_grad = False

    def _init_params(self):
        # optional load pretrained weights from matconvnet
        if (self.matconvnet is not None):
            self.base.load_state_dict(torch.load(self.matconvnet))
            self.pretrained = True

    def forward(self, x):
        N,C,H,W = x.size()
        
        if C == 4:
            x = x[:,:3,:,:]  # extract rgb
        # return middle layer
        returnVector = None
        if self.return_layer != None:
            if self.return_layer == "conv6":
                x = self.base(x)
                returnVector = x
            else:
                returnVector = x
                for i in range(VGG.__fix_layers[self.return_layer]+1):
                    returnVector = self.base[i](returnVector)
                x = returnVector
                for i in range(VGG.__fix_layers[self.return_layer]+1, len(self.base)):
                    x = self.base[i](x)
        else:
            x = self.base(x)

        if self.cut_at_pooling:
            return x.view(x.size(0), -1) if self.isflatten==True else x
        
        pool_x = x
        if self.gap != None:
            pool_x = self.gap(pool_x)
            pool_x = pool_x.view(pool_x.size(0), -1)
            pool_x = F.normalize(pool_x, p=2, dim=1)  # L2 normalize

        # for guiuded
        if self.return_layer != None and self.gap == None:
            # print(returnVector.shape, x.shape)
            return returnVector, x
        elif self.return_layer != None and self.gap != None:
            return pool_x, returnVector
        
        return pool_x, x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def vgg16(**kwargs):
    return VGG(16, **kwargs)
