from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import random
import numpy as np
import sys
import json

import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

from flashtorch.activmax import GradientAscent  # for visualiz

sys.path.append('../')
from ibl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict, write_json

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from ibl.utils.data import get_transformer_test

class VGG(nn.Module):
    __factory = {
        16: torchvision.models.vgg16,
    }
    __fix_layers = { # vgg16
        28:'conv5',
        21:'conv4',
        14:'conv3',
        7: 'conv2',
        2: 'conv1'
    }
    def __init__(self, depth, pretrained=True, matconvnet=None):
        super(VGG, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.feature_dim = 512
        self.matconvnet = matconvnet
        self.fix_layers = VGG.__fix_layers
        if depth not in VGG.__factory:
            raise KeyError("Unsupported depth:", depth) 
        vgg = VGG.__factory[depth](pretrained=pretrained)
        # layers = list(vgg.features.children())[:-2]
        layers = list(vgg.features.children())
        self.base = nn.Sequential(*layers) # capture only feature part and remove last relu and maxpool
        self._init_params()
    def _init_params(self):
        # optional load pretrained weights from matconvnet
        if (self.matconvnet is not None):
            # self.base.load_state_dict(torch.load(self.matconvnet))
            self.pretrained = True
    def forward(self, x):
        N,C,H,W = x.size()
        # return middle layer
        returnVector = []
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in VGG.__fix_layers.keys():
                returnVector.append((VGG.__fix_layers[i], x))
        return returnVector, x

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""
    def __init__(self, num_clusters=64, dim=512, alpha=100.0, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim), requires_grad=True)

        self.clsts = None
        self.traindescs = None
    def _init_params(self):
        clstsAssign = self.clsts / np.linalg.norm(self.clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, self.traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids.data.copy_(torch.from_numpy(self.clsts))
        self.conv.weight.data.copy_(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
    def forward(self, x):
        N, C, H, W = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        returnSoft = soft_assign.view(N, self.num_clusters, H, W)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters in one loop
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        returnResidual = residual
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        return returnSoft, returnResidual, vlad

class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()
    
    def forward(self, x):
        N,C,H,W = x.size()
        returnlayers, x = self.base_model(x)
        returnSoft, returnResidual, vlad_x = self.net_vlad(x)
        # returnlayers.append(("softAssign", returnSoft))
        # returnlayers.append(("returnResidual", returnResidual))

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        return returnlayers, vlad_x

def get_model():
    base_model = VGG(16, pretrained=True, matconvnet='../../logs/vd16_offtheshelf_conv5_3_max.pth')
    pool_layer = NetVLAD(num_clusters=64, dim=512, alpha=100.0, normalize_input=True)
    model = EmbedNet(base_model, pool_layer)

    model.cuda()
    model = nn.DataParallel(model, device_ids=[0], output_device=0)
    return model

def load_ckpt(model, ckpt_path):
    # Load from checkpoint
    checkpoint = load_checkpoint(ckpt_path)
    copy_state_dict(checkpoint['state_dict'], model)
    start_epoch = checkpoint['epoch']
    best_recall5 = checkpoint['best_recall5']
    best_gen = checkpoint['generation'] if "generation" in checkpoint.keys() else None
    print("=> Start gen {} epoch {}  best recall5 {:.1%}"
        .format(best_gen, start_epoch, best_recall5))

def get_attentionMaps(attentionMaps, visType="abs_sum", height=480, width=640):
    """
    Visualize intermediate feature maps

    Args:
        attentionMaps (Dict): {"name1 (str)": "array (Torch.tensor)", ...}, array's size is (1,D,W,H)
        visType (Str): visualized method
        height (Int): the height of original image
        width (Int): the width of original image
    Return:
        images (List): attention maps
    """
    images = []
    for name, infeatures in attentionMaps:
        # infeature is (B,D,W,H)
        if visType == "max":
            infeatures = torch.abs(infeatures)
            infeatures,_ = torch.max(infeatures, dim=1)
        elif visType == "pow2":
            infeatures = infeatures.pow(2).mean(1)
        elif visType == "abs_sum":
            infeatures = torch.abs(infeatures)
            infeatures = infeatures.mean(1)
        
        # resize成原图片大小
        infeatures = infeatures.unsqueeze(1)  # (B, 1, H, W)
        infeatures = torch.nn.functional.interpolate(infeatures, (height, width), mode="bilinear", align_corners=False)
        infeatures = infeatures.squeeze().cpu().numpy()

        # 初始化归一化模板
        normalizer = mpl.colors.Normalize(vmin=infeatures.min(), vmax=infeatures.max())
        # 该方法用于scalar data to RGBA mapping，用于可视化
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
        colormapped_im = (mapper.to_rgba(infeatures)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        images.append(im)
    return images


def visualize_attentionMaps(imagesShow, img_path):
    """
    Save visualized images with multi rows, one row-[original image, visualized_image0, visualized_image1, visualized_image2, ...]

    Args:
        imagesShow (List): images to be saved
        image_path (Str): output path
    """
    rows = len(imagesShow)
    cols = len(imagesShow[0])
    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(3.2*cols, 2.4*rows))
    for row in range(rows):
        for col in range(cols):
            ax.ravel()[row*cols+col].imshow(imagesShow[row][col])
            # print(row, col)
            ax.ravel()[row*cols+col].set_axis_off()
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize feature maps')
    parser.add_argument('--resume', type=str, default="../../logs/netvlad/pitts30k-vgg16/conv5-triplet-lr0.001-neg1-tuple4/model_best.pth.tar")
    parser.add_argument('--imagePath', type=str, default="./images/test0.jpg")
    parser.add_argument('--outputPath', type=str, default="./results/test0_visualized.jpg")
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)

    args = parser.parse_args()

    # Get model
    model = get_model()

    # Load ckpt
    load_ckpt(model, args.resume)
    # save_checkpoint(model.module.base_model.base.state_dict(), is_best=False, fpath="../../logs/vgg16_netvlad.pth")

    # Pre-process image
    transformer = get_transformer_test(args.height, args.width)
    image = pil.open(args.imagePath).convert('RGB')

    # Inference
    with torch.no_grad():
        returnlayers, vlad_x = model(transformer(image).unsqueeze(0))

    # Visualize
    imagesShow = []
    returnImages = get_attentionMaps(returnlayers, visType="abs_sum", height=args.height, width=args.width)
    imagesShow.append([image] + returnImages)
    visualize_attentionMaps(imagesShow, args.outputPath)
    