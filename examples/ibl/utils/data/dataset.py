from __future__ import print_function
import math
import os.path as osp
import numpy as np
import copy
import torch.distributed as dist
from sklearn.neighbors import NearestNeighbors

from ..serialization import read_json, read_mat

def minDeltaDegree(degree1, degree2):
    return min(np.abs(degree1 - degree2), np.abs(min(degree1, degree2)+360-max(degree1, degree2)))

def _pluck(identities, utm, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for fname in pid_images:
            x, y = utm[pid]
            if relabel:
                ret.append((fname, index, x, y))
            else:
                ret.append((fname, pid, x, y))
    return sorted(ret)

def _pluck_uacLike(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for fname in pid_images:
            x = -1
            y = -1
            if relabel:
                ret.append((fname, index, x, y))
            else:
                ret.append((fname, pid, x, y))
    return ret

def _pluck_robotcarLike(identities, utm, degrees, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for fname in pid_images:
            x, y = utm[pid]
            degree = degrees[pid]
            if relabel:
                ret.append((fname, index, x, y, degree))
            else:
                ret.append((fname, pid, x, y, degree))
    return sorted(ret)

def get_groundtruth_robotcarLike(query, gallery, intra_degree, intra_thres, inter_thres=None):
    utm_query = [[u[2], u[3]] for u in query]
    utm_gallery = [[u[2], u[3]] for u in gallery]

    neigh = NearestNeighbors(n_jobs=-1)
    neigh.fit(utm_gallery)
    # test the distance
    dist, neighbors = neigh.radius_neighbors(utm_query, radius=intra_thres)
    pos, select_pos = [], []
    for idx, p in enumerate(neighbors):
        pid = query[idx][1]
        # test the degree and exclude itself
        select_p = [i for i in p.tolist() if (gallery[i][1]!=pid and minDeltaDegree(query[idx][4], gallery[i][4])<intra_degree)]
        if (len(select_p)>0):
            pos.append(select_p)
            select_pos.append(idx)
    if (inter_thres is None):
        return pos, select_pos
    dist, neighbors = neigh.radius_neighbors(utm_query, radius=inter_thres)
    neg = [n.tolist() for n in neighbors]
    return pos, neg, select_pos

def get_groundtruth(query, gallery, intra_thres, inter_thres=None):
    utm_query = [[u[2], u[3]] for u in query]
    utm_gallery = [[u[2], u[3]] for u in gallery]
    neigh = NearestNeighbors(n_jobs=-1)
    neigh.fit(utm_gallery)
    dist, neighbors = neigh.radius_neighbors(utm_query, radius=intra_thres)
    pos, select_pos = [], []
    for idx, p in enumerate(neighbors):
        pid = query[idx][1]
        select_p = [i for i in p.tolist() if gallery[i][1]!=pid]
        if (len(select_p)>0):
            pos.append(select_p)
            select_pos.append(idx)
    if (inter_thres is None):
        return pos, select_pos
    dist, neighbors = neigh.radius_neighbors(utm_query, radius=inter_thres)
    neg = [n.tolist() for n in neighbors]
    return pos, neg, select_pos

class Dataset(object):
    def __init__(self, root, usedegree=False, intra_thres=10, inter_thres=25):
        self.root = root
        self.usedegree = usedegree
        self.intra_thres = intra_thres
        self.inter_thres = inter_thres
        self.train = []
        self.q_val, self.db_val = [], []
        self.q_test, self.db_test = [], []
        self.train_pos, self.train_neg, self.val_pos, self.val_neg, \
                self.test_pos, self.test_neg = [], [], [], [], [], []

    @property
    def images_dir(self):
        return osp.join(self.root, 'raw')

    def load(self, verbose, scale=None):
        if (scale is None):
            splits = read_json(osp.join(self.root, 'splits.json'))
            meta = read_json(osp.join(self.root, 'meta.json'))
        else:
            splits = read_json(osp.join(self.root, 'splits_'+scale+'.json'))
            meta = read_json(osp.join(self.root, 'meta_'+scale+'.json'))
        identities = meta['identities']
        utm = meta['utm']

        q_train_pids = sorted(splits['q_train'])
        db_train_pids = sorted(splits['db_train'])
        train_pids = q_train_pids + db_train_pids
        q_val_pids = sorted(splits['q_val'])
        db_val_pids = sorted(splits['db_val'])
        q_test_pids = sorted(splits['q_test'])
        db_test_pids = sorted(splits['db_test'])

        self.q_train = _pluck(identities, utm, q_train_pids, relabel=False)
        self.db_train = _pluck(identities, utm, db_train_pids, relabel=False)
        
        self.train = self.q_train + self.db_train
        self.q_val = _pluck(identities, utm, q_val_pids, relabel=False)
        self.db_val = _pluck(identities, utm, db_val_pids, relabel=False)
        self.q_test = _pluck(identities, utm, q_test_pids, relabel=False)
        self.db_test = _pluck(identities, utm, db_test_pids, relabel=False)

        self.train_pos, self.train_neg, select = get_groundtruth(self.q_train, self.db_train, self.intra_thres, self.inter_thres)
        self.train_neg = [self.train_neg[idx] for idx in select]  # 这句话有问题，如此不能选到25m内的样本, 就不应该加这句
        self.q_train = [self.q_train[idx] for idx in select]  # 只选择有正样本对应的qeury样本
        q_train_pids = list(set([x[1] for x in self.q_train]))
        db_train_pids = list(set([x[1] for x in self.db_train]))

        self.val_pos, select = get_groundtruth(self.q_val, self.db_val, 25, None)  # in xxm is positive
        self.q_val = [self.q_val[idx] for idx in select]  # 我加的, 为了筛选出可以匹配的样本
        assert(len(select)==len(self.q_val))
        self.test_pos, select = get_groundtruth(self.q_test, self.db_test, 25, None)
        self.q_test = [self.q_test[idx] for idx in select]  # 我加的, 为了筛选出可以匹配的样本
        assert(len(select)==len(self.q_test))

        try:
            rank = dist.get_rank()
        except:
            rank = 0
        if (verbose and rank==0):
            print(self.__class__.__name__, "dataset loaded")
            print("  subset        | # pids | # images")
            print("  ---------------------------------")
            print("  train_query   | {:5d}  | {:8d}"
                  .format(len(q_train_pids), len(self.q_train)))
            print("  train_gallery | {:5d}  | {:8d}"
                  .format(len(db_train_pids), len(self.db_train)))
            print("  val_query     | {:5d}  | {:8d}"
                  .format(len(q_val_pids), len(self.q_val)))
            print("  val_gallery   | {:5d}  | {:8d}"
                  .format(len(db_val_pids), len(self.db_val)))
            print("  test_query    | {:5d}  | {:8d}"
                  .format(len(q_test_pids), len(self.q_test)))
            print("  test_gallery  | {:5d}  | {:8d}"
                  .format(len(db_test_pids), len(self.db_test)))

    def load_uacLike(self, verbose):
        splits = read_json(osp.join(self.root, 'splits.json'))
        meta = read_json(osp.join(self.root, 'meta.json'))
        identities = meta['identities']

        q_train_pids = splits['q_train']
        db_train_pids = splits['db_train']
        train_pids = q_train_pids + db_train_pids
        q_val_pids = splits['q_val']
        db_val_pids = splits['db_val']
        q_test_pids = splits['q_test']
        db_test_pids = splits['db_test']
        # print(identities)

        self.q_train = _pluck_uacLike(identities, q_train_pids, relabel=False)
        self.db_train = _pluck_uacLike(identities, db_train_pids, relabel=False)
        self.train = self.q_train + self.db_train
        self.q_val = _pluck_uacLike(identities, q_val_pids, relabel=False)
        self.db_val = _pluck_uacLike(identities, db_val_pids, relabel=False)
        self.q_test = _pluck_uacLike(identities, q_test_pids, relabel=False)
        self.db_test = _pluck_uacLike(identities, db_test_pids, relabel=False)

        assert len(self.q_val)== len(self.q_test)
        select_pos = []
        for _gt in meta["gt"]:
            select_pos.append(_gt)
        self.val_pos = select_pos
        self.test_pos = self.val_pos
        try:
            rank = dist.get_rank()
        except:
            rank = 0
        
        if (verbose and rank==0):
            print(self.__class__.__name__, "dataset loaded")
            print("  subset        | # pids | # images")
            print("  ---------------------------------")
            print("  train_query   | {:5d}  | {:8d}"
                  .format(len(q_train_pids), len(self.q_train)))
            print("  train_gallery | {:5d}  | {:8d}"
                  .format(len(db_train_pids), len(self.db_train)))
            print("  val_query     | {:5d}  | {:8d}"
                  .format(len(q_val_pids), len(self.q_val)))
            print("  val_gallery   | {:5d}  | {:8d}"
                  .format(len(db_val_pids), len(self.db_val)))
            print("  test_query    | {:5d}  | {:8d}"
                  .format(len(q_test_pids), len(self.q_test)))
            print("  test_gallery  | {:5d}  | {:8d}"
                  .format(len(db_test_pids), len(self.db_test)))


    def _check_integrity(self, scale=None):
        if (scale is None):
            return osp.isfile(osp.join(self.root, 'meta.json')) and \
                   osp.isfile(osp.join(self.root, 'splits.json'))
        else:
            return osp.isfile(osp.join(self.root, 'meta_'+scale+'.json')) and \
                   osp.isfile(osp.join(self.root, 'splits_'+scale+'.json'))


class DatasetRobotcar(object):
    def __init__(self, root, intra_degree=30, intra_thres=5, inter_thres=25):
        self.root = root
        self.intra_thres = intra_thres
        self.inter_thres = inter_thres
        self.intra_degree = intra_degree
        self.train = []
        self.q_val, self.db_val = [], []
        self.q_test, self.db_test = [], []
        self.train_pos, self.train_neg, self.val_pos, self.val_neg, \
                self.test_pos, self.test_neg = [], [], [], [], [], []

    @property
    def images_dir(self):
        return self.root
    
    def load(self, verbose, scale=None):
        if (scale is None):
            splits = read_json(osp.join(self.root, 'splits.json'))
            meta = read_json(osp.join(self.root, 'meta.json'))
        else:
            splits = read_json(osp.join(self.root, 'splits_'+scale+'.json'))
            meta = read_json(osp.join(self.root, 'meta_'+scale+'.json'))
        identities = meta['identities']
        utm = meta['utm']
        degrees = meta['degrees']

        q_train_pids = sorted(splits['q_train'])
        db_train_pids = sorted(splits['db_train'])
        train_pids = q_train_pids + db_train_pids
        q_val_pids = sorted(splits['q_val'])
        db_val_pids = sorted(splits['db_val'])
        q_test_pids = sorted(splits['q_test'])
        db_test_pids = sorted(splits['db_test'])

        self.q_train = _pluck_robotcarLike(identities, utm, degrees, q_train_pids, relabel=False)
        self.db_train = _pluck_robotcarLike(identities, utm, degrees, db_train_pids, relabel=False)
        
        self.train = self.q_train + self.db_train
        self.q_val = _pluck_robotcarLike(identities, utm, degrees, q_val_pids, relabel=False)
        self.db_val = _pluck_robotcarLike(identities, utm, degrees, db_val_pids, relabel=False)
        self.q_test = _pluck_robotcarLike(identities, utm, degrees, q_test_pids, relabel=False)
        self.db_test = _pluck_robotcarLike(identities, utm, degrees, db_test_pids, relabel=False)

        self.train_pos, self.train_neg, select = get_groundtruth_robotcarLike(self.q_train, self.db_train, self.intra_degree, self.intra_thres, self.inter_thres)
        # print(len(self.train_neg), self.train_neg[0])
        self.train_neg = [self.train_neg[idx] for idx in select]  # 只选择有意义query样本的对应负样本
        self.q_train = [self.q_train[idx] for idx in select]  # 只选择有正样本对应的qeury样本
        q_train_pids = list(set([x[1] for x in self.q_train]))
        db_train_pids = list(set([x[1] for x in self.db_train]))

        self.val_pos, select = get_groundtruth(self.q_val, self.db_val, 25, None)  # in xxm is positive
        self.q_val = [self.q_val[idx] for idx in select]  # 我加的, 为了筛选出可以匹配的样本
        assert(len(select)==len(self.q_val))
        self.test_pos, select = get_groundtruth(self.q_test, self.db_test, 25, None)
        self.q_test = [self.q_test[idx] for idx in select]  # 我加的, 为了筛选出可以匹配的样本
        assert(len(select)==len(self.q_test))

        try:
            rank = dist.get_rank()
        except:
            rank = 0
        if (verbose and rank==0):
            print(self.__class__.__name__, "dataset loaded")
            print("  subset        | # pids | # images")
            print("  ---------------------------------")
            print("  train_query   | {:5d}  | {:8d}"
                    .format(len(q_train_pids), len(self.q_train)))
            print("  train_gallery | {:5d}  | {:8d}"
                    .format(len(db_train_pids), len(self.db_train)))
            print("  val_query     | {:5d}  | {:8d}"
                    .format(len(q_val_pids), len(self.q_val)))
            print("  val_gallery   | {:5d}  | {:8d}"
                    .format(len(db_val_pids), len(self.db_val)))
            print("  test_query    | {:5d}  | {:8d}"
                    .format(len(q_test_pids), len(self.q_test)))
            print("  test_gallery  | {:5d}  | {:8d}"
                    .format(len(db_test_pids), len(self.db_test)))

    def _check_integrity(self, scale=None):
        if (scale is None):
            return osp.isfile(osp.join(self.root, 'meta.json')) and \
                   osp.isfile(osp.join(self.root, 'splits.json'))
        else:
            return osp.isfile(osp.join(self.root, 'meta_'+scale+'.json')) and \
                   osp.isfile(osp.join(self.root, 'splits_'+scale+'.json'))

class DatasetRobotcarSemantics(object):
    def __init__(self, root, intra_degree=30, intra_thres=5, inter_thres=25):
        self.root = root
        self.intra_thres = intra_thres
        self.inter_thres = inter_thres
        self.intra_degree = intra_degree
        self.train = []
        self.q_val, self.db_val = [], []
        self.q_test, self.db_test = [], []
        self.train_pos, self.train_neg, self.val_pos, self.val_neg, \
                self.test_pos, self.test_neg = [], [], [], [], [], []

    @property
    def images_dir(self):
        return self.root
    
    def load(self, verbose, scale=None):
        if (scale is None):
            splits = read_json(osp.join(self.root, 'splitsSemantics.json'))
            meta = read_json(osp.join(self.root, 'metaSemantics.json'))
        else:
            splits = read_json(osp.join(self.root, 'splitsSemantics_'+scale+'.json'))
            meta = read_json(osp.join(self.root, 'metaSemantics_'+scale+'.json'))
        identities = meta['identities']
        utm = meta['utm']
        degrees = meta['degrees']

        q_train_pids = sorted(splits['q_train'])
        db_train_pids = sorted(splits['db_train'])
        train_pids = q_train_pids + db_train_pids
        q_val_pids = sorted(splits['q_val'])
        db_val_pids = sorted(splits['db_val'])
        q_test_pids = sorted(splits['q_test'])
        db_test_pids = sorted(splits['db_test'])

        self.q_train = _pluck_robotcarLike(identities, utm, degrees, q_train_pids, relabel=False)
        self.db_train = _pluck_robotcarLike(identities, utm, degrees, db_train_pids, relabel=False)
        
        self.train = self.q_train + self.db_train
        self.q_val = _pluck_robotcarLike(identities, utm, degrees, q_val_pids, relabel=False)
        self.db_val = _pluck_robotcarLike(identities, utm, degrees, db_val_pids, relabel=False)
        self.q_test = _pluck_robotcarLike(identities, utm, degrees, q_test_pids, relabel=False)
        self.db_test = _pluck_robotcarLike(identities, utm, degrees, db_test_pids, relabel=False)

        self.train_pos, self.train_neg, select = get_groundtruth_robotcarLike(self.q_train, self.db_train, self.intra_degree, self.intra_thres, self.inter_thres)
        self.train_neg = [self.train_neg[idx] for idx in select]  # 只选择有意义query样本的对应负样本
        self.q_train = [self.q_train[idx] for idx in select]  # 只选择有正样本对应的qeury样本
        q_train_pids = list(set([x[1] for x in self.q_train]))
        db_train_pids = list(set([x[1] for x in self.db_train]))

        self.val_pos, select = get_groundtruth(self.q_val, self.db_val, 25, None)  # in xxm is positive
        self.q_val = [self.q_val[idx] for idx in select]  # 我加的, 为了筛选出可以匹配的样本
        assert(len(select)==len(self.q_val))
        self.test_pos, select = get_groundtruth(self.q_test, self.db_test, 25, None)
        self.q_test = [self.q_test[idx] for idx in select]  # 我加的, 为了筛选出可以匹配的样本
        assert(len(select)==len(self.q_test))

        try:
            rank = dist.get_rank()
        except:
            rank = 0
        if (verbose and rank==0):
            print(self.__class__.__name__, "dataset loaded")
            print("  subset        | # pids | # images")
            print("  ---------------------------------")
            print("  train_query   | {:5d}  | {:8d}"
                    .format(len(q_train_pids), len(self.q_train)))
            print("  train_gallery | {:5d}  | {:8d}"
                    .format(len(db_train_pids), len(self.db_train)))
            print("  val_query     | {:5d}  | {:8d}"
                    .format(len(q_val_pids), len(self.q_val)))
            print("  val_gallery   | {:5d}  | {:8d}"
                    .format(len(db_val_pids), len(self.db_val)))
            print("  test_query    | {:5d}  | {:8d}"
                    .format(len(q_test_pids), len(self.q_test)))
            print("  test_gallery  | {:5d}  | {:8d}"
                    .format(len(db_test_pids), len(self.db_test)))

    def _check_integrity(self, scale=None):
        if (scale is None):
            return osp.isfile(osp.join(self.root, 'metaSemantics.json')) and \
                   osp.isfile(osp.join(self.root, 'splitsSemantics.json'))
        else:
            return osp.isfile(osp.join(self.root, 'metaSemantics_'+scale+'.json')) and \
                   osp.isfile(osp.join(self.root, 'splitsSemantics_'+scale+'.json'))