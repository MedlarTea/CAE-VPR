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
import os

def load_gt(gt_csv, q_nums, db_nums):
    tol = 50
    gt_csv = pd.read_csv(gt_csv, header=None)
    gt_list = []
    for index, row in gt_csv.iterrows():
        gt = row[0]
        if gt<tol:
            gt = np.arange(1, gt+tol)
        elif gt>=tol and gt<db_nums-tol:
            gt = np.arange(gt-tol, gt+tol)
        else:
            gt = np.arange(gt-tol, db_nums) 
        gt_list.append(gt.tolist())
    print(len(gt_list), q_nums, db_nums)
    assert len(gt_list) == q_nums
    return gt_list

class Alderley(Dataset):
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
        super(Alderley, self).__init__(root)

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

        db_dir = osp.join(self.root, 'FRAMESA')
        query_dir = osp.join(self.root, 'FRAMESB')
        gt_csv = osp.join(self.root, 'framematches.csv')
        
        query_image_paths = []
        query_img_nums = len(os.listdir(query_dir))
        for i in range(1, query_img_nums+1):
            num = str(i).zfill(5)
            image_path = osp.join(query_dir, 'Image{}.jpg'.format(num))
            if not osp.isfile(image_path):
                raise RuntimeError("No image: {}".format(image_path))
            query_image_paths.append([image_path])
        
        db_image_paths = []
        db_img_nums = len(os.listdir(db_dir))
        for i in range(1, db_img_nums+1):
            num = str(i).zfill(5)
            image_path = osp.join(db_dir, 'Image{}.jpg'.format(num))
            if not osp.isfile(image_path):
                raise RuntimeError("No image: {}".format(image_path))
            db_image_paths.append([image_path])
        
        identities = []
        q_train_pids = []
        db_train_pids = []
        q_val_pids = []
        db_val_pids = []
        q_test_pids = []
        db_test_pids = []

        q_train_pids += [i for i in range(len(identities), len(identities)+query_img_nums)]
        identities += query_image_paths

        db_train_pids += [i for i in range(len(identities), len(identities)+db_img_nums)]
        identities += db_image_paths


        q_val_pids += [i for i in range(len(identities), len(identities)+query_img_nums)]
        identities += query_image_paths

        db_val_pids += [i for i in range(len(identities), len(identities)+db_img_nums)]
        identities += db_image_paths


        q_test_pids += [i for i in range(len(identities), len(identities)+query_img_nums)]
        identities += query_image_paths

        db_test_pids += [i for i in range(len(identities), len(identities)+db_img_nums)]
        identities += db_image_paths

        # print(identities)
        gt = load_gt(gt_csv, query_img_nums, db_img_nums)
        # if rank == 0:
        #     print(gt)
        meta = {'name': 'Alderley',
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
