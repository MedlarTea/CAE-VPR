from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import random
import numpy as np
import sys

import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

sys.path.append('../')
from ibl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict, write_json
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

from sklearn.metrics import pairwise_distances


class VGG(nn.Module):
    __factory = {
        16: torchvision.models.vgg16,
    }
    __fix_layers = { # vgg16
        28:'conv6',
        24:'conv5',
        17:'conv4',
        10:'conv3',
        5: 'conv2'
    }
    def __init__(self, depth, pretrained=True, matconvnet=None):
        super(VGG, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.feature_dim = 512
        self.matconvnet = matconvnet
        if depth not in VGG.__factory:
            raise KeyError("Unsupported depth:", depth) 
        vgg = VGG.__factory[depth](pretrained=pretrained)
        layers = list(vgg.features.children())[:-2]
        self.base = nn.Sequential(*layers) # capture only feature part and remove last relu and maxpool
        self._init_params()
    def _init_params(self):
        # optional load pretrained weights from matconvnet
        if (self.matconvnet is not None):
            self.base.load_state_dict(torch.load(self.matconvnet))
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
    print("=> Start epoch {}  best recall5 {:.1%}"
        .format(start_epoch, best_recall5))

def get_image(image_path):
    # Load image and transform it
    img = pil.open(image_path).convert('RGB') # modify the image path according to your need
    transformer = transforms.Compose([transforms.Resize((480, 640)), # (height, width)
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                                        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
    img = transformer(img)
    return img

def visualize(infeatures, result_path, returnlayer="conv5", visType="pow2", height=480, width=640):
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
    
    for index, infeature in enumerate(infeatures):
        infeature = infeature.squeeze().cpu().numpy()  # (B, H, W)
        print("returnlayer: {}, min: {:.3f}, max: {:.3f}, mean: {:.3f}".format(returnlayer, infeature.min(), infeature.max(), infeature.mean()))
        # 初始化归一化模板
        normalizer = mpl.colors.Normalize(vmin=infeature.min(), vmax=infeature.max())
        # 该方法用于scalar data to RGBA mapping，用于可视化
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
        colormapped_im = (mapper.to_rgba(infeature)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        # store path
        basename = osp.basename(result_path).strip('.jpg') + "_{}_{}.jpg".format(returnlayer, visType)
        store_path = osp.join(osp.dirname(result_path), basename)
        im.save(store_path)

def draw_distribution(matches, distri_curve_path):
    """评估query-database中, 模型对于正样本和负样本的区分度
    
    Args:
        matches: n个欧氏距离
        distri_curve_path: 分布图存储位置
    
    Returns:
        存储分布直方图以及它们的均值方差
    """
    plt.figure()
    mean = np.mean(matches)
    std = np.var(matches)
    counts_x, bins_x = np.histogram(matches,bins=20)
    plt.hist(bins_x[:-1], bins_x, weights=counts_x/len(matches), color="darkorange", edgecolor = 'black', alpha=0.5, label="matches(u={:.3f}, std={:.3f})".format(mean, std))
    plt.legend()
    plt.xlabel('L2 Distance')
    plt.ylabel('Probability')
    plt.title('Netvlad')
    plt.savefig(distri_curve_path)
    plt.close()

def get_mask(semantic_path, image):
    semantic_mask = pil.open(semantic_path)

    transformer = transforms.Compose([transforms.Resize((480,640)), transforms.ToTensor()])

    semantic_mask = transformer(semantic_mask)
    semantic_mask[semantic_mask < 3] = 0
    semantic_mask[semantic_mask > 9] = 0
    semantic_mask[semantic_mask != 0] = 1
    filtered_image = torch.mul(semantic_mask, image)
    return filtered_image

if __name__ == '__main__':
    # Get model
    model = get_model()
    # Load ckpt
    # ckpt_path = "/home/jing/Models/OpenIBL/logs/netVLAD/pitts30k-vgg16/conv5-triplet-lr0.001-neg5-tuple4/model_best.pth.tar"
    # ckpt_path = "/home/jing/Models/OpenIBL/logs/netVLAD/pitts30k-vgg16/author/conv5-sare_ind-lr0.001-tuple4-SFRS/model_best.pth.tar"
    ckpt_path = "/home/lab/data1/hanjingModel/OpenIBL_forRobotcar/logs/netVLAD/attention/teacher/robotcar-vgg16/conv5-triplet-lr0.001-neg10-tuple2/model_best.pth.tar"
    load_ckpt(model, ckpt_path)

    # Visualize the Att map
    # Load image and transform it
    image_path = "/home/lab/data1/hanjing/singleframe/Kudamm_RAS2020/Live/image0024.jpg"
    result_path = "./visualized/image0024.jpg"
    img = get_image(image_path)

    # extract descriptor (4096-dim)
    with torch.no_grad():
        returnlayers, vlad_x = model(img.unsqueeze(0))
    
    
    # visualize
    for name, feature in returnlayers:
        visualize(feature, result_path, returnlayer=name, visType="max", height=480, width=640)
    

    # Draw the L2 Distribution of Dynamic Environment
    # images_dir = "/home/jing/Data/Dataset/Street/CityStreet/CityStreet_zip/CityStreet/image_frames/camera1"
    # semantics_dir = images_dir + "_s"
    # vladList = []
    # for image in os.listdir(images_dir):
    #     image_path = osp.join(images_dir, image)
    #     semantics_path = osp.join(semantics_dir, image.split('.')[0]+"_mask.png")
    #     img = get_image(image_path)
    #     img = get_mask(semantics_path, img)  # filter

    #     with torch.no_grad():
    #         _, output = model(img.unsqueeze(0))
    #     vladList.append(output.squeeze().cpu().numpy())
    # dist = pairwise_distances(vladList)

    # _dist = []
    # for i in range(dist.shape[0]-1):
    #     for j in range(i+1, dist.shape[1]):
    #         _dist.append(dist[i][j])
    
    # distri_curve_path = osp.join(osp.dirname(images_dir), "camera1_distri.png")
    # draw_distribution(_dist, distri_curve_path)
    
    


    