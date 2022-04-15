import os.path as osp
import pandas as pd
import torch.distributed as dist

from collections import namedtuple
from ..utils.data import Dataset
from ..utils.serialization import write_json
from ..utils.dist_utils import synchronize

import numpy as np
from scipy.spatial.distance import cdist,cosine
import scipy.io as scio

def load_gt(gt_file):
    gt = scio.loadmat(gt_file)
    gt = gt["UAcampus"]
    gt = np.array(gt, dtype=np.uint)
    img_nums = 647
    gt = gt[:img_nums, :img_nums]

    gt_list=[]  # (img_nums, [1,2,3,...])
    for i in range(img_nums):
        gt_list.append(np.where(gt[i]==1)[0].tolist())
    return gt_list

class Uacampus(Dataset):
    """
    examples/data
    └── demo
        ├── raw/
        ├── meta.json
        └── splits.json

    Inputs:
        root (str): the path to demo_dataset
        verbose (bool): print flag, default=True
    """

    def __init__(self, root, verbose=True):
        super(Uacampus, self).__init__(root)

        self.query_img_nums = 647
        self.db_img_nums = 647

        self.arrange()
        self.load_uacLike(verbose)

    def arrange(self):
        # if self._check_integrity():
        #     return

        try:
            rank = dist.get_rank()
        except:
            rank = 0

        # the root path for raw dataset
        if (not osp.isdir(self.root)):
            print(self.root)
            raise RuntimeError("Dataset not found.")

        query_dir = osp.join(self.root, 'query/images')
        db_dir = osp.join(self.root, 'database/images')
        gt_mat = osp.join(self.root, 'gt.mat')
        
        query_image_paths = []
        for i in range(self.query_img_nums):
            query_image_paths.append([osp.join(query_dir, '{}.jpg'.format(i))])
        
        db_image_paths = []
        for i in range(self.db_img_nums):
            db_image_paths.append([osp.join(db_dir, '{}.jpg'.format(i))])
        
        identities = []
        q_train_pids = []
        db_train_pids = []
        q_val_pids = []
        db_val_pids = []
        q_test_pids = []
        db_test_pids = []

        q_train_pids += [i for i in range(len(identities), len(identities)+self.query_img_nums)]
        identities += query_image_paths

        db_train_pids += [i for i in range(len(identities), len(identities)+self.db_img_nums)]
        identities += db_image_paths


        q_val_pids += [i for i in range(len(identities), len(identities)+self.query_img_nums)]
        identities += query_image_paths

        db_val_pids += [i for i in range(len(identities), len(identities)+self.db_img_nums)]
        identities += db_image_paths


        q_test_pids += [i for i in range(len(identities), len(identities)+self.query_img_nums)]
        identities += query_image_paths

        db_test_pids += [i for i in range(len(identities), len(identities)+self.db_img_nums)]
        identities += db_image_paths

        # print(identities)
        gt = load_gt(gt_mat)
        meta = {'name': 'Uacampus',
            'identities': identities, 'gt': gt}
        try:
            rank = dist.get_rank()
        except:
            rank = 0
        if rank == 0:
            write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the training / test split
        splits = {
            # 'train': sorted(train_pids),
            'q_train': q_train_pids,
            'db_train': db_train_pids,
            'q_val': q_val_pids,
            'db_val': db_val_pids,
            'q_test': q_test_pids,
            'db_test': db_test_pids}
        if rank == 0:
            write_json(splits, osp.join(self.root, 'splits.json'))
        synchronize()
