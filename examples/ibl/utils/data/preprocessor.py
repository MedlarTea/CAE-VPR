from __future__ import absolute_import
# from examples.ibl.utils import data
import os
import re
import os.path as osp
import numpy as np
import random
import math
from PIL import Image
import cv2
import json

import torch
# from torch._C import float32
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms.transforms import RandomRotation

class Preprocessor_fusion(Dataset):
    def __init__(self, dataset, root=None, useSemantics=False, useDepth=False, height=480, width=640, isRobotcar=False):
        super(Preprocessor_fusion, self).__init__()
        self.dataset = dataset
        self.root = root
        self.useSemantics = useSemantics
        self.useDepth = useDepth
        self.height = height
        self.width = width
        self.isRobotcar = isRobotcar

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def getOtherPath(self, img, tmp_path_list, _type):
        if _type == "semantic":
            if tmp_path_list[-3] == "images":
                tmp_path_list[-3] = 'images_semantics'
            elif tmp_path_list[-3] == "queries":
                tmp_path_list[-3] = 'queries_semantics'
            if tmp_path_list[-2] == "centre":
                tmp_path_list[-2] = "semantic"
            semantic_path = osp.join(self.root, '/'.join(tmp_path_list))
            semantic = Image.open(semantic_path)
            semantic = preprocess_semantic(semantic, self.height, self.width)
            return torch.cat((img, semantic), 0)
        if _type == "depth":
            if tmp_path_list[-3] == "images":
                tmp_path_list[-3] = 'images_depth'
            elif tmp_path_list[-3] == "queries":
                tmp_path_list[-3] = 'queries_depth'
            if tmp_path_list[-2] == "centre":
                tmp_path_list[-2] = "depth"
            depth_path = osp.join(self.root, '/'.join(tmp_path_list))
            depth = Image.open(depth_path)
            depth = preprocess_depth(depth, self.height, self.width)
            return torch.cat((img, depth), 0)
        return img

    def _get_single_item(self, index):
        if self.isRobotcar == True:
            fname, pid, x, y, _ = self.dataset[index]
        else:
            fname, pid, x, y = self.dataset[index]
        fpath = fname 
        if self.root is not None:
            # for Robotcar, root/stereo/centre/xxx.jpg
            fpath = osp.join(self.root, fname)  # (/home/jk/hanjing/Models/OpenIBL/examples/data/pitts, raw/Pittsburgh/images/000/xxx.jpg)
            
            img = Image.open(fpath).convert('RGB')
            img = preprocess_image(img, self.height, self.width)
            tmp_path_list = fname.split('/')
            if self.useSemantics:
                img = self.getOtherPath(img, tmp_path_list, _type="semantic")
            if self.useDepth:
                img = self.getOtherPath(img, tmp_path_list, _type="depth")
        return img, fname, pid, x, y

class Preprocessor(Dataset):
    __semantic={"road":0, "sidewalk":1, "building":2, "wall":3, "fence":4, "pole":5, "light":6, "traffic-sign":7, "vegetation":8, "terrain":9, "sky":10, "dynamic objects":[x for x in range(11,19)]}
    def __init__(self, dataset, root=None, transform=None, useSemantics=False, height=480, width=640, isRobotcar=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.useSemantics = useSemantics
        self.isRobotcar = isRobotcar
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        if self.isRobotcar == True:
            fname, pid, x, y, _ = self.dataset[index]
        else:
            fname, pid, x, y = self.dataset[index]
        fpath = fname 
        if self.root is not None:
            fpath = osp.join(self.root, fname)  # (/home/jk/hanjing/Models/OpenIBL/examples/data/pitts, raw/Pittsburgh/images/000/xxx.jpg)
            img = Image.open(fpath).convert('RGB')
            if (self.transform is not None):
                img = self.transform(img)
            if self.useSemantics:
                semantic_path_list = fname.split('/')

                if semantic_path_list[-3] == "images":
                    semantic_path_list[-3] = 'images_semantics'
                elif semantic_path_list[-3] == "queries":
                    semantic_path_list[-3] = 'queries_semantics'
                if semantic_path_list[-2] == "centre":
                    semantic_path_list[-2] = "semantic"
                

                semantic_path = osp.join(self.root, '/'.join(semantic_path_list))
                semantic = Image.open(semantic_path)
                semantic = filter_semantic(semantic, self.height, self.width)
                img = torch.cat((img, semantic), 0)
        return img, fname, pid, x, y

def filter_semantic(semantic, height, width):
    r'''Preprocess the semantic mask

        The pitts' semantic mask is inferred from deeplabv3plus trained in cityscapes. Then transform [0,18] of uint8 to 0/1 mask of torch.float32
        We set 0, 1, 10, 11-18 to 0, others are 1

        Input:
            PIL.Image size(W,H)/(1,W,H) in range[0,18]
            with 0-road, 1-sidewalk, 2-building, 3-wall, 4-fence, 5-pole, 6-light, 7-traffic-sign, 8-vegetation, 9-terrain, 10-sky, 11-18-dynamic objects

        Output:
            (1,H,W) of torch.float32 with 0/1 mask
    '''
    semantic = np.array(semantic)
    semantic[semantic<0]=0
    semantic[semantic>18]=18
    semantic[semantic < 2] = 99  # 移除road相关
    semantic[semantic > 9] = 99  # 移除sky和动态物体
    semantic[semantic == 8] = 99  # 移除vegetation
    semantic[semantic != 99] = 1
    semantic[semantic == 99] = 0

    semantic = Image.fromarray(semantic)
    transformer = T.Compose([T.Resize((height,width)), T.ToTensor()])
    semantic = transformer(semantic)
    
    return semantic

def preprocess_semantic(semantic, height, width):
    r'''Preprocess the semantic mask

        The pitts' semantic mask is inferred from deeplabv3plus trained in cityscapes. Then transform [0,18] of uint8 to 0/1 mask of torch.float32
        We set 0, 1, 10, 11-18 to 0, others are 1

        Input:
            PIL.Image size(W,H)/(1,W,H) in range[0,18]
            with 0-road, 1-sidewalk, 2-building, 3-wall, 4-fence, 5-pole, 6-light, 7-traffic-sign, 8-vegetation, 9-terrain, 10-sky, 11-18-dynamic objects

        Output:
            (1,H,W) of torch.float32 with 0/1 mask
    '''
    transformer = T.Compose([T.Resize((height,width))])
    semantic = transformer(semantic)
    semantic = np.array(semantic)
    semantic[semantic<0]=0
    semantic[semantic>18]=18
    semantic = semantic / 18
    semantic = torch.from_numpy(semantic).unsqueeze(0)
    return semantic

def encode_semantic(semantic, height, width):
    r'''Use for semantic segmentation
        Input:
            PIL.Image size(W,H)/(1,W,H) in range[0,18]
            with 0-road, 1-sidewalk, 2-building, 3-wall, 4-fence, 5-pole, 6-light, 7-traffic-sign, 8-vegetation, 9-terrain, 10-sky, 11-18-dynamic objects
    '''
    transformer = T.Compose([T.Resize((height,width))])
    semantic = transformer(semantic)
    semantic = np.array(semantic)
    # print(semantic.max(), semantic.min())
    semantic[semantic<0]=0
    semantic[semantic>18]=255
    # print(semantic.max(), semantic.min())
    semantic = torch.from_numpy(semantic).unsqueeze(0)
    return semantic
def preprocess_depth(depth, height, width):
    '''
        depth is [0, 255]
    '''
    transformer = T.Compose([T.Resize((height,width)), T.ToTensor()]) 
    depth = transformer(depth)
    return depth

def preprocess_image(img, height, width):
    '''
        image is [0, 255]
    '''
    transformer = T.Compose([T.Resize((height,width)),
                             T.ColorJitter(0.7, 0.7, 0.7, 0.5),
                             T.ToTensor()])
    img = transformer(img)
    return img

def changeName(fpath, replaceType=None):
    if replaceType is None:
        return 
    if replaceType == "semantic":
        fpath = fpath.replace("images", 'images_semantics')
        # for the queries of the pitts
        fpath = fpath.replace("queries", 'queries_semantics')
        fpath = fpath.replace("centre", 'semantic')
    elif replaceType == "depth":
        # for the database of the pitts
        fpath = fpath.replace("images", 'depth')
        # for the queries of the pitts
        fpath = fpath.replace("queries", 'queries_depth')
        # for the robotcar
        fpath = fpath.replace("centre", 'depth')
        # for the queries of the singleFrame
        fpath = fpath.replace("Live", 'Live_depth')
        # for the database of the singleFrame
        fpath = fpath.replace("Reference", 'Reference_depth')
    elif replaceType == "saliency":
        # for the database of the pitts
        fpath = fpath.replace("images", 'images_saliency')
        # for the queries of the pitts
        fpath = fpath.replace("queries", 'queries_saliency')
        # for the robotcar
        fpath = fpath.replace("centre", 'saliency')
        # for the queries of the singleFrame
        fpath = fpath.replace("Live", 'Live_saliency')
        # for the database of the singleFrame
        fpath = fpath.replace("Reference", 'Reference_saliency')
    return fpath

class PreprocessDisentangle(Dataset):
    __semantic={"road":0, "sidewalk":1, "building":2, "wall":3, "fence":4, "pole":5, "light":6, "traffic-sign":7, "vegetation":8, "terrain":9, "sky":10, "dynamic objects":[x for x in range(11,19)]}
    def __init__(self, dataset, root=None, transform=None, useSemantics=False, height=480, width=640, isRobotcar=False):
        super(PreprocessDisentangle, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.useSemantics = useSemantics
        self.isRobotcar = isRobotcar
        self.height = height
        self.width = width
        self.filtered_classes = ["building","wall","fence","pole","light","traffic-sign","terrain"]  # important objects

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        if self.isRobotcar == True:
            fname, pid, x, y, _ = self.dataset[index]
        else:
            fname, pid, x, y = self.dataset[index]
        fpath = fname 
        if self.root is not None:
            fpath = osp.join(self.root, fname)  # (/home/jk/hanjing/Models/OpenIBL/examples/data/pitts, raw/Pittsburgh/images/000/xxx.jpg)
            img = Image.open(fpath).convert('RGB')
            if (self.transform is not None):
                img = self.transform(img)
            if self.useSemantics:
                semantic_path_list = fname.split('/')

                if semantic_path_list[-3] == "images":
                    semantic_path_list[-3] = 'images_semantics'
                elif semantic_path_list[-3] == "queries":
                    semantic_path_list[-3] = 'queries_semantics'
                if semantic_path_list[-2] == "centre":
                    semantic_path_list[-2] = "semantic"
                

                semantic_path = osp.join(self.root, '/'.join(semantic_path_list))
                semantic = Image.open(semantic_path)
                semantic = self.get_semantic_masks(semantic)
                img = torch.cat((img, semantic), 0)
        return img, fname, pid, x, y

    def get_semantic_masks(self, semantic):
        transformer = T.Compose([T.Resize((self.height,self.width))])
        semantic = transformer(semantic)
        semantic = np.array(semantic)
        masks = []
        mask = np.zeros(semantic.shape, dtype=np.int)
        for _class in self.filtered_classes:
            mask[semantic==PreprocessDisentangle.__semantic[_class]] = 1
        mask = torch.from_numpy(mask)
        masks.append(mask)
        if len(masks)==1:
            return masks[0].unsqueeze(0)
        else:
            return torch.stack(masks, dim=0)

class NormalLoading(Dataset):
    def __init__(self, dataset, root=None, height=480, width=640, isRobotcar=False, useSemantic=False, useDepth=False):
        super(NormalLoading, self).__init__()
        self.dataset = dataset
        self.root = root
        self.height = height
        self.width = width
        self.isRobotcar = isRobotcar
        self.useSemantic = useSemantic
        self.useDepth = useDepth
        # from Robotcar, rgbsd
        self.mean = [0.4864, 0.4904, 0.5544, 0.2419, 0.2471]
        self.std = [0.2967, 0.3000, 0.3000, 0.2536, 0.2040]
        # from mapillary, rgbsd
        # self.mean = [0.4158, 0.4519, 0.4601, 0., 0.3085]
        # self.std = [0.2460, 0.2603, 0.2843, 1., 0.2088]
        self.normalize = self.get_normalize_transformer(useSemantic = useSemantic, useDepth = useDepth)  # if we use depth as ground truth, do not normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)
    
    def _get_single_item(self, index):
        if self.isRobotcar == True:
            fname, pid, x, y, _ = self.dataset[index]
        else:
            fname, pid, x, y = self.dataset[index]
        fpath = fname 
        if self.root is not None:
            fpath = osp.join(self.root, fname)  # (/home/jk/hanjing/Models/OpenIBL/examples/data/pitts, raw/Pittsburgh/images/000/xxx.jpg)
            img = Image.open(fpath).convert('RGB')
            img = preprocess_image(img, self.height, self.width)
            # print(img.shape) 
            if self.useSemantic:
                semantic_path = changeName(fpath, replaceType="semantic")
                semantic = Image.open(semantic_path)
                # semantic = preprocess_semantic(semantic, self.height, self.width)
                semantic = encode_semantic(semantic, self.height, self.width)  # for semantic segmention
                img = torch.cat((img, semantic), 0)
            if self.useDepth:
                depth_path = changeName(fpath, replaceType="depth")
                depth = Image.open(depth_path)
                depth = preprocess_depth(depth, self.height, self.width)
                img = torch.cat((img, depth), 0)
            # print(img[3,:,:].max(), img[3,:,:].min())
            # print(img.shape)
            img = self.normalize(img)
            # print(img[3,:,:].max(), img[3,:,:].min())
        return img, fname, pid, x, y
    
    def get_normalize_transformer(self, useSemantic=False, useDepth=False):
        if not useSemantic and not useDepth:
            mean = self.mean[:3]
            std = self.std[:3]
        elif useSemantic and not useDepth:
            mean = self.mean[:4]
            std = self.std[:4]
        elif not useSemantic and useDepth:
            # mean = [0.,0.,0.] + [self.mean[4]]
            # std = [1.,1.,1.] + [self.std[4]]
            mean = self.mean[:3] + [self.mean[4]]
            std = self.std[:3] + [self.std[4]]
        else:
            # mean = self.mean
            # std = self.std
            mean = self.mean[:3] + [0.] + [self.mean[4]]
            std = self.std[:3] + [1.] + [self.std[4]]
        # for unsupervised learning
        # return T.Compose([T.RandomHorizontalFlip(),
        #                   T.RandomRotation(20),
        #                   T.Normalize(mean=mean,
        #                               std=std)])
        # for supervised learning
        return T.Compose([T.Normalize(mean=mean,
                                      std=std)])

class SaliencyPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, height=480, width=640, isRobotcar=False):
        super(SaliencyPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.isRobotcar = isRobotcar
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        if self.isRobotcar == True:
            fname, pid, x, y, _ = self.dataset[index]
        else:
            fname, pid, x, y = self.dataset[index]
        fpath = fname 
        if self.root is not None:
            fpath = osp.join(self.root, fname)  # (/home/jk/hanjing/Models/OpenIBL/examples/data/pitts, raw/Pittsburgh/images/000/xxx.jpg)
            saliency_path = changeName(fpath, replaceType="saliency")
            saliency_path = os.path.splitext(saliency_path)[0] + ".json"
            img = Image.open(fpath).convert('RGB')
            masks = self.load_saliency(saliency_path)
            if (self.transform is not None):
                img = self.transform(img)
                img = torch.cat((img, masks), 0)
        return img, fname, pid, x, y
    
    def load_saliency(self, json_file):
        """
            return saliency masks with 1x12xHxW dimensions
        """
        width = 640
        height = 480
        with open(json_file) as f:
            saliency = json.load(f)
        boxes = saliency["boxes"][:11]
        N = saliency["numbers"]
        # if N!=12:
        #     print(N)
        #     print(json_file)
            # assert N==12
        # resize the boxes, to match the size of conv5's output
        # and change to masks
        masks = torch.zeros([11, height, width])
        for i, box in enumerate(boxes):
            masks[i, box[1]:box[3],box[0]:box[2]] = 1
        return masks
    
class SSLPreprocessor(Dataset):
    def __init__(self, datasets, indices, root=None, pt_method="moco", transform=None):
        super(SSLPreprocessor, self).__init__()
        # here, we use mapillary as our unsupervised leanring dataset
        # it has multi cities
        self.datasets = datasets
        self.indices = indices
        self.root = root
        self.pt_method = pt_method
        self.transform = transform

    def getInterval(self, index):
        start = 0
        end = 0
        for i, indice in enumerate(self.indices):
            if index >= indice:
                start = indice
                end = self.indices[i+1]
            else:
                return (start, end)
        return (start, end)

    def __len__(self):
        return len(self.datasets)

    def get_moco_items(self, index):
        # return query, aug_query, negative
        # MENTION: query and negative come from different cities
        neg_index = index
        start, end = self.getInterval(index)
        while neg_index<end and neg_index>=start:
            neg_index = random.randint(1, self.__len__()-1)
        q_fname, pid, x, y = self.datasets[index]
        neg_fname, pid, x, y = self.datasets[neg_index]
        if self.root is not None:
            q_fpath = osp.join(self.root, q_fname)
            neg_path = osp.join(self.root, neg_fname)
            q_img = Image.open(q_fpath).convert('RGB')
            pos_img = q_img
            neg_img = Image.open(neg_path).convert('RGB')
            if (self.transform is not None):
                q_img = self.transform(q_img)
                pos_img = self.transform(pos_img)
                neg_img = self.transform(neg_img)
        return q_img, pos_img, neg_img

    def __getitem__(self, index):
        if  self.pt_method=="moco":
            q_img, pos_img, neg_img = self.get_moco_items(index)
            return [q_img, pos_img, neg_img], index    
    