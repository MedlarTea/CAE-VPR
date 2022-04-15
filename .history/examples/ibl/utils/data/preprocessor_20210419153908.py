from __future__ import absolute_import
import os
import re
import os.path as osp
import numpy as np
import random
import math
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, useSemantics=False, isRobotcar=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.useSemantics = useSemantics
        self.isRobotcar = isRobotcar

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
                semantic_mask = Image.open(semantic_path)
                semantic_mask = preprocess_semantic(semantic_mask)
                img = torch.cat((img, semantic_mask), 0)
        return img, fname, pid, x, y

# def preprocess_semantic(semantic_mask):
#     r'''Preprocess the semantic mask

#         The semantic mask is inferred from deeplabv3plus trained in cityscapes. Then transform [0,18] of uint8 to [0,1] of torch.float32  
#         Input:
#             PIL.Image size(W,H)/(1,W,H) in range[0,18]
#             with 0-road, 1-sidewalk, 2-building, 3-wall, 4-fence, 5-pole, 6-light, 7-traffic-sign, 8-vegetation, 9-terrain, 10-sky, 11-18-dynamic objects
#         Output:
#             (1,H,W) of torch.float32 in range[0,1]
#     '''
#     return (T.ToTensor()(semantic_mask)+(1/255)) * 255 / 19

def preprocess_semantic(semantic_mask):
    r'''Preprocess the semantic mask

        The pitts' semantic mask is inferred from deeplabv3plus trained in cityscapes. Then transform [0,18] of uint8 to 0/1 mask of torch.float32
        We set 0, 1, 10, 11-18 to 0, others are 1

        Input:
            PIL.Image size(W,H)/(1,W,H) in range[0,18]
            with 0-road, 1-sidewalk, 2-building, 3-wall, 4-fence, 5-pole, 6-light, 7-traffic-sign, 8-vegetation, 9-terrain, 10-sky, 11-18-dynamic objects

        Output:
            (1,H,W) of torch.float32 with 0/1 mask
    '''
    transformer = T.Compose([T.Resize((480,640))])
    semantic_mask = transformer(semantic_mask)
    semantic_mask = torch.tensor(np.array(semantic_mask))
    # semantic_mask[semantic_mask < 3] = 20  # 可能路还是很重要的
    semantic_mask[semantic_mask > 9] = 99  # 移除sky和动态物体
    semantic_mask[semantic_mask != 99] = 1
    semantic_mask[semantic_mask == 99] = 0
    semantic_mask = semantic_mask.unsqueeze(0)
    assert semantic_mask.size()[0] == 1
    
    return semantic_mask