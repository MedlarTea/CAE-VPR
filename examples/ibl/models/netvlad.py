import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
from .models_utils import *
import torch.nn.functional as nnf

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

        # semantics and rgb fusion branch
        self.conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), bias=False)

    def _init_params(self):
        clstsAssign = self.clsts / np.linalg.norm(self.clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, self.traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids.data.copy_(torch.from_numpy(self.clsts))
        self.conv.weight.data.copy_(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))

    def forward(self, x, semantic=None):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        # soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # soft_assign = F.softmax(soft_assign, dim=1)

        # for semantic and rgb fusion
        if semantic != None:
            soft_assign = self.conv(x)
            soft_assign = torch.cat((soft_assign, semantic), 1)
            soft_assign = self.conv1(soft_assign).view(N, self.num_clusters, -1)
            soft_assign = F.softmax(soft_assign, dim=1)
        else:
            soft_assign = self.conv(x).view(N, self.num_clusters, -1)  # (bs, K, W*H)
            soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters in one loop
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        return vlad

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
        # print(x.size())
        pool_x, x = self.base_model(x)
        vlad_x = self.net_vlad(x) 

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        return pool_x, vlad_x

class RGBD2SEmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad, net_semantic=None):
        super(RGBD2SEmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.net_semantic = net_semantic

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        rgb = x[:,:3,:,:]
        depth = x[:,3,:,:].unsqueeze(1)
        pool_x, x = self.base_model(rgb, depth)
        vlad_x = self.net_vlad(x) 

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        return pool_x, vlad_x


class EmbedRGBSNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedRGBSNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        N,C,H,W = x.size()
        # print(x.size())
        if C == 4:
            encoded, _ = self.base_model(x)
            vlad_x = self.net_vlad(encoded)
        else:
            raise("Error!")

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        return encoded, vlad_x

class EmbedNetPCA(nn.Module):
    def __init__(self, base_model, net_vlad, dim=4096):
        super(EmbedNetPCA, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.pca_layer = nn.Conv2d(net_vlad.num_clusters*net_vlad.dim, dim, 1, stride=1, padding=0)

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        _, x = self.base_model(x)
        vlad_x = self.net_vlad(x)

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        # reduction
        N, D = vlad_x.size()
        vlad_x = vlad_x.view(N, D, 1, 1)
        vlad_x = self.pca_layer(vlad_x).view(N, -1)
        vlad_x = F.normalize(vlad_x, p=2, dim=-1)  # L2 normalize

        return vlad_x

class EmbedRegionNet(nn.Module):
    def __init__(self, base_model, net_vlad, tuple_size=1):
        super(EmbedRegionNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.tuple_size = tuple_size

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def _compute_region_sim(self, feature_A, feature_B):
        # feature_A: B*C*H*W
        # feature_B: (B*(1+neg_num))*C*H*W

        def reshape(x):
            # re-arrange local features for aggregating quarter regions
            N, C, H, W = x.size()
            x = x.view(N, C, 2, int(H/2), 2, int(W/2))
            x = x.permute(0,1,2,4,3,5).contiguous()
            x = x.view(N, C, -1, int(H/2), int(W/2))
            return x

        feature_A = reshape(feature_A)
        feature_B = reshape(feature_B)

        # computer quarter-region features
        def aggregate_quarter(x):
            N, C, B, H, W = x.size()
            x = x.permute(0,2,1,3,4).contiguous()
            x = x.view(-1,C,H,W)
            vlad_x = self.net_vlad(x) # (N*B)*64*512
            _, cluster_num, feat_dim = vlad_x.size()
            vlad_x = vlad_x.view(N,B,cluster_num,feat_dim)
            return vlad_x

        vlad_A_quarter = aggregate_quarter(feature_A)
        vlad_B_quarter = aggregate_quarter(feature_B)

        # computer half-region features
        def quarter_to_half(vlad_x):
            return torch.stack((vlad_x[:,0]+vlad_x[:,1], vlad_x[:,2]+vlad_x[:,3], \
                                vlad_x[:,0]+vlad_x[:,2], vlad_x[:,1]+vlad_x[:,3]), dim=1).contiguous()

        vlad_A_half = quarter_to_half(vlad_A_quarter)
        vlad_B_half = quarter_to_half(vlad_B_quarter)

        # computer global-image features
        def quarter_to_global(vlad_x):
            return vlad_x.sum(1).unsqueeze(1).contiguous()

        vlad_A_global = quarter_to_global(vlad_A_quarter)
        vlad_B_global = quarter_to_global(vlad_B_quarter)

        def norm(vlad_x):
            N, B, C, _ = vlad_x.size()
            vlad_x = F.normalize(vlad_x, p=2, dim=3)  # intra-normalization
            vlad_x = vlad_x.view(N, B, -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # L2 normalize
            return vlad_x

        vlad_A = torch.cat((vlad_A_global, vlad_A_half, vlad_A_quarter), dim=1)
        vlad_B = torch.cat((vlad_B_global, vlad_B_half, vlad_B_quarter), dim=1)
        vlad_A = norm(vlad_A)
        vlad_B = norm(vlad_B)

        _, B, L = vlad_B.size()
        vlad_A = vlad_A.view(self.tuple_size,-1,B,L)
        vlad_B = vlad_B.view(self.tuple_size,-1,B,L)

        score = torch.bmm(vlad_A.expand_as(vlad_B).contiguous().view(-1,B,L), vlad_B.contiguous().view(-1,B,L).transpose(1,2))
        score = score.view(self.tuple_size,-1,B,B)

        return score, vlad_A, vlad_B

    def _forward_train(self, x):
        B, C, H, W = x.size()
        x = x.view(self.tuple_size, -1, C, H, W)

        anchors = x[:, 0].unsqueeze(1).contiguous().view(-1,C,H,W) # B*C*H*W
        pairs = x[:, 1:].contiguous().view(-1,C,H,W) # (B*(1+neg_num))*C*H*W

        return self._compute_region_sim(anchors, pairs)

    def forward(self, x):
        pool_x, x = self.base_model(x)

        if (not self.training):
            vlad_x = self.net_vlad(x)
            # normalize
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
            vlad_x = vlad_x.view(x.size(0), -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize
            return pool_x, vlad_x

        return self._forward_train(x)

class SemanticNet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SemanticNet, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        filters = [64, 128, 256, 512]
        # encoder
        self.conv1 = create_conv_1(self.input_channel, filters[0])
        self.conv2 = create_conv_1(filters[0], filters[1])
        self.conv3 = create_conv_1(filters[1], filters[2])
        self.conv4 = create_conv_1(filters[2], filters[3])
        self.conv5 = nn.Conv2d(filters[3], self.output_channel, kernel_size=1, bias=False)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        return x

class VggConvAuto(nn.Module):
    def __init__(self, base_model, convAuto_model, islayerNorm=False):
        super(VggConvAuto, self).__init__()
        self.base_model = base_model
        self.convAuto_model = convAuto_model
        self.islayerNorm = islayerNorm
        if self.islayerNorm:
            self.layernorm = nn.LayerNorm([512, 30, 40], elementwise_affine=False)

        self._init_params()

    def _init_params(self):
        self.base_model._init_params()
        # 冻结vgg参数
        layers = list(self.base_model.children())
        for l in layers:
            for p in l.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.base_model(x)
        if self.islayerNorm:
            features = self.layernorm(features)
        encoded, decoded = self.convAuto_model(features)
        return features, encoded, decoded

class AlexnetConvAuto(nn.Module):
    def __init__(self, base_model, convAuto_model, islayerNorm=False):
        super(AlexnetConvAuto, self).__init__()
        self.base_model = base_model
        self.convAuto_model = convAuto_model
        self.islayerNorm = islayerNorm
        if self.islayerNorm:
            self.layernorm = nn.LayerNorm([256, 28, 38], elementwise_affine=False)

        self._init_params()

    def _init_params(self):
        self.base_model._init_params()
        # 冻结vgg参数
        layers = list(self.base_model.children())
        for l in layers:
            for p in l.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.base_model(x)
        if self.islayerNorm:
            features = self.layernorm(features)
        encoded, decoded = self.convAuto_model(features)
        return features, encoded, decoded

class EmbedAttentionNet(nn.Module):
    def __init__(self, base_model, net_vlad, useSemantics=False, isvlad=False, visType='abs_sum', sigma=.5, w=8):
        super(EmbedAttentionNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.useSemantics = useSemantics
        self.visType = visType
        self.isvlad = isvlad
        self.sigma = sigma
        self.w = w

    def _init_params(self):
        self.base_model._init_params()
        if self.net_vlad is not None:
            self.net_vlad._init_params()

    def forward(self, x):
        N,C,H,W = x.size()
        # rgb = x[:,:3,:,:]  # extract rgb
        # semantic = x[:,3,:,:]  # semantic mask--0/1 mask with (B, H, W)
        if self.useSemantics and self.net_vlad is not None:
            att_mask, x = self.base_model(x)
            vlad_x = self.net_vlad(x)
            att_mask = self.get_mask(att_mask, height=H, width=W, visType=self.visType, sigma=self.sigma, w=self.w)
        elif self.useSemantics and self.net_vlad is None:
            x, att_mask = self.base_model(x)
            att_mask = self.get_mask(att_mask, height=H, width=W, visType=self.visType, sigma=self.sigma, w=self.w)
            return x, att_mask
        else:
            att_mask, x = self.base_model(x)
            vlad_x = self.net_vlad(x) 

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        return att_mask, vlad_x
    
    def get_mask(self, infeatures, height=480, width=640, visType='abs_sum', sigma=.5, w=8):
        # infeature is (B,D,W,H)
        B, _ , _, _ = infeatures.size()
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
        infeatures = infeatures.squeeze()  # (B, H, W)

        # 标准化 + softmax
        infeatures = infeatures.view(infeatures.size(0), -1) 
        infeatures -= infeatures.min(1, keepdim=True)[0]
        infeatures /= infeatures.max(1, keepdim=True)[0]
        infeatures = infeatures.view(B, height, width)
        mask = torch.sigmoid(w * (infeatures - sigma))

        return mask

class DisentangleNet(nn.Module):
    def __init__(self, base_model, net_vlad, fcae, useSemantics=False):
        super(DisentangleNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.fcae = fcae
        self.useSemantics = useSemantics

        self._init_params()

    def _init_params(self):
        for i, (name, param) in enumerate(self.base_model.named_parameters()):
            param.requires_grad = False
        for i, (name, param) in enumerate(self.net_vlad.named_parameters()):
            param.requires_grad = False
    
    def _forward_train(self, features, semantic):
        # features: N, C, H, W
        # semantic: N, class, H, W (it's a mask with 0/1)

        def reshape(x):
            # flatten local features to extract 1x1 feature
            # print(x.size())
            N, C, H, W = x.size()
            x = x.view(N, C, -1, 1, 1)  # 1x1 patch, N*C*(H*W)*1*1
            return x
        def reshape_semantic(x, height, width):
            x =nnf.interpolate(x, size=(height, width), mode='nearest')
            N, C, H, W = x.size()
            x = x.view(N, C, -1, 1, 1) # bs, class, h*w, 1, 1
            return x

        
        
        _,_,h,w = features.size()
        features = reshape(features)
        semantic = reshape_semantic(semantic,h,w)
        
        # compute 1x1 features
        def aggregate_single(x):
            N, C, B, H, W = x.size()
            x = x.permute(0,2,1,3,4).contiguous()
            x = x.view(-1,C,H,W)
            x = self.net_vlad(x) # (N*B)*64*512
            _, cluster_num, feat_dim = x.size()
            x = x.view(N,1,B,cluster_num,feat_dim)  # N,1,B,cluster_num,feat_dim
            return x
        
        features = aggregate_single(features)

        # compute region features
        def single_to_region(x, s):
            # x is N,1,B,cluster_num,feat_dim
            # s is N, C, B, 1, 1
            x = torch.mul(x, s) # N, C, 1, cluster_num,feat_dim
            x = x.sum(2).squeeze(2).contiguous() # N,C,cluster_num,feat_dim
            return x

        # compute global-image features
        def single_to_global(x):
            return x.sum(2).contiguous()  # N,1,cluster_num,feat_dim
        
        features_region = single_to_region(features, semantic)
        features_global = single_to_global(features)
        

        def norm(x):
            N, B, _, _ = x.size()
            x = F.normalize(x, p=2, dim=3)  # intra-normalization
            x = x.view(N, B, -1)  # flatten
            x = F.normalize(x, p=2, dim=2)  # L2 normalize
            return x
        # print(features_global.shape)
        # print(features_region.shape)
        
        features_region = norm(features_region)
        features_global = norm(features_global)

        def flatten(x):
            N,_,_ = x.size()
            return x.view(N,-1)
        
        features_global = flatten(features_global)
        features_global = self.fcae(features_global)
        output = torch.cat((features_global.unsqueeze(1), features_region), dim=1)  # N,(1+C),cluster_num*feat_dim
        
        return output

    def forward(self, x):
        N,C,H,W = x.size()
        if self.useSemantics:
            semantic = x[:,3:,:,:]
            # print(semantic.shape)
            x = x[:,:3,:,:]
            # print(x.shape)
        if (not self.training):
            _, x = self.base_model(x)
            vlad_x = self.net_vlad(x)
            # normalize
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
            vlad_x = vlad_x.view(N, -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize
            vlad_x = self.fcae(vlad_x)
            return vlad_x
        else:
            _, x = self.base_model(x)
            return self._forward_train(x, semantic)

