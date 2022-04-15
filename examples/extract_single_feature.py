from __future__ import print_function, absolute_import
import os
import time
import argparse
import string
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.nn import Parameter
from sklearn.metrics import average_precision_score

import cv2
import _pickle as cPickle

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.feature_dim = 512
        vgg = torchvision.models.vgg16(pretrained=True)
        layers = list(vgg.features.children())[:-2]
        self.base = nn.Sequential(*layers) # capture only feature part and remove last relu and maxpool
    def forward(self, x):
        # s1 = time.time()
        N,C,H,W = x.size()
        x = self.base(x)   
        # print("VGG inference: {:.3f}".format(time.time()-s1))
        return x

class convAuto(nn.Module):
    def __init__(self, d1,d2,dimension):
        super(convAuto, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(512, d1, (4,4), stride=(1,1), padding=0),
            nn.BatchNorm2d(d1),
            nn.PReLU(),

            nn.Conv2d(d1, d2, (7,5), stride=(2,2), padding=0), 
            nn.BatchNorm2d(d2),
            nn.PReLU(),

            nn.Conv2d(d2, dimension, (5,3), stride=(2,2), padding=0),   # dimension x 4 x 8
            nn.BatchNorm2d(dimension),
            nn.PReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, d2, (5,3), stride=(2,2), padding=0), 
            nn.BatchNorm2d(d2),
            nn.PReLU(),

            nn.ConvTranspose2d(d2, d1, (7,5), stride=(2,2), padding=0), 
            nn.BatchNorm2d(d1),
            nn.PReLU(),

            nn.ConvTranspose2d(d1, 512, (4,4), stride=(1,1), padding=0), 
            nn.BatchNorm2d(512),
            nn.PReLU()
            # nn.Tanh()
        )
    def forward(self,x):
        # s1 = time.time()
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        # print("convAuto inference: {:.3f}".format(time.time()-s1))
        return x

class VggConvAuto(nn.Module):
    def __init__(self, base_model, convAuto_model, islayerNorm=False):
        super(VggConvAuto, self).__init__()
        self.base_model = base_model
        self.convAuto_model = convAuto_model
        self.islayerNorm = islayerNorm
        if self.islayerNorm:
            self.layernorm = nn.LayerNorm([512, 30, 40], elementwise_affine=False)

    def forward(self, x):
        features = self.base_model(x)
        if self.islayerNorm:
            features = self.layernorm(features)
        encoded = self.convAuto_model(features)
        # return encoded
        return F.normalize(encoded, p=2, dim=-1)
        # return encoded


def copy_state_dict(state_dict, model, strip=None, replace=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        # print(name)
        if replace is not None and name.find(replace[0]) != -1:
            name = name.replace(replace[0], replace[1])
        # print(name)
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)
    missing = set(tgt_state.keys()) - copied_names
    if ((len(missing) > 0)):
        print("missing keys in state_dict:", missing)
    return model

def get_transformer_test(height, width, tokyo=False):
    test_transformer = [T.Resize(max(height,width) if tokyo else (height, width)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                   std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])]
    return T.Compose(test_transformer)

def get_model(args):
    base_model = VGG()
    convAuto_model = convAuto(d1=args.d1, d2=args.d2, dimension=args.dimension)
    model = VggConvAuto(base_model, convAuto_model, islayerNorm=True)
    return model

def main():
    args = parser.parse_args()
    # Create model
    model = get_model(args)
    model.cuda()

    # Load from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        copy_state_dict(checkpoint['state_dict'], model, strip=None, replace=["module.", ""])
        start_epoch = checkpoint['epoch']
        best_recall5 = checkpoint['best_recall5']
        print("=> Start epoch {}  best recall5 {:.1%}"
                .format(start_epoch, best_recall5))
    
    # get data
    img_path = args.img_path
    img = Image.open(img_path)
    img_transformer = get_transformer_test(480, 640)
    img = img_transformer(img).unsqueeze(0).cuda()
    with torch.no_grad():
        desc = model(img)
        print("desc's dimension: {}".format(desc.shape))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")

    parser.add_argument('--d1', type=int, default=128)
    parser.add_argument('--d2', type=int, default=128)
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--resume', type=str, default='logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-d1-128-d2-128-dimension1024/checkpoint49.pth.tar', metavar='PATH')
    parser.add_argument('--img_path', type=str, default="")
    main()
    
    


