import os.path as osp
import pandas as pd
import torch.distributed as dist

from collections import namedtuple
from ..utils.data import Dataset
from ..utils.serialization import write_json
from ..utils.dist_utils import synchronize

def read_csv(image_dir, dataframe):
    image_names = []
    utms = []
    for index, row in dataframe.iterrows():
        image_names.append([osp.join(image_dir, row["key"]+".jpg")])
        utms.append([row["easting"], row["northing"]])
    return image_names, utms, len(image_names)

def parse_dataset(image_dir, gt_csv, isUseNight=True):
    df = pd.read_csv(gt_csv)
    if isUseNight==False:
        trainSet = df[df["night"]==False]
    else:
        trainSet = df

    train_image_names, trainUtms, train_nums = read_csv(image_dir, trainSet)
    valid_image_names, validUtms, valid_nums = (train_image_names, trainUtms, train_nums)
    test_image_names, testUtms, test_nums = (train_image_names, trainUtms, train_nums)
    dbStruct = namedtuple('dbStruct',['train_image_names', 'trainUtms', 'train_nums', 
                                        'valid_image_names', 'validUtms', 'valid_nums',
                                        'test_image_names', 'testUtms', 'test_nums'])
    return dbStruct(train_image_names, trainUtms, train_nums, 
                    valid_image_names, validUtms, valid_nums,
                    test_image_names, testUtms, test_nums)

class Mapillary(Dataset):
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

    def __init__(self, root, city='train_val/amman', isUseNight=True, verbose=True):
        super(Mapillary, self).__init__(root)
        self.city = city
        self.isUseNight = isUseNight
        self.root = osp.join(self.root, self.city)

        self.arrange()
        self.load(verbose)

    def arrange(self):
        if self._check_integrity():
            return

        try:
            rank = dist.get_rank()
        except:
            rank = 0

        query_dir = osp.join(self.root, 'query')
        query_image_dir = osp.join(query_dir, 'images')
        query_raw_gt = osp.join(query_dir, 'raw.csv')
        query_gt = osp.join(query_dir, 'postprocessed.csv')

        db_dir = osp.join(self.root, 'database')
        db_image_dir = osp.join(db_dir, 'images')
        db_raw_gt = osp.join(db_dir, 'raw.csv')
        db_gt = osp.join(db_dir, 'postprocessed.csv')

        if (not osp.isdir(self.root)):
            raise RuntimeError("Dataset not found.")
        
        identities = []
        utms = []
        q_train_pids = []
        db_train_pids = []
        q_val_pids = []
        db_val_pids = []
        q_test_pids = []
        db_test_pids = []

        struct_query = parse_dataset(query_image_dir, query_gt, self.isUseNight)
        struct_db = parse_dataset(db_image_dir, db_gt, self.isUseNight)

        q_train_pids += [i for i in range(len(identities), len(identities)+struct_query.train_nums)]
        identities += struct_query.train_image_names
        utms += struct_query.trainUtms
        db_train_pids += [i for i in range(len(identities), len(identities)+struct_db.train_nums)]
        identities += struct_db.train_image_names
        utms += struct_db.trainUtms

        q_val_pids += [i for i in range(len(identities), len(identities)+struct_query.valid_nums)]
        identities += struct_query.valid_image_names
        utms += struct_query.validUtms
        db_val_pids += [i for i in range(len(identities), len(identities)+struct_db.valid_nums)]
        identities += struct_db.valid_image_names
        utms += struct_db.validUtms

        q_test_pids += [i for i in range(len(identities), len(identities)+struct_query.test_nums)]
        identities += struct_query.test_image_names
        utms += struct_query.testUtms
        db_test_pids += [i for i in range(len(identities), len(identities)+struct_db.test_nums)]
        identities += struct_db.test_image_names
        utms += struct_db.testUtms

        assert len(identities)==len(utms)
        meta = {'name': 'Mapillary-{}'.format(self.city),
            'identities': identities, 'utm': utms}
        try:
            rank = dist.get_rank()
        except:
            rank = 0
        if rank == 0:
            write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the training / test split
        splits = {
            # 'train': sorted(train_pids),
            'q_train': sorted(q_train_pids),
            'db_train': sorted(db_train_pids),
            'q_val': sorted(q_val_pids),
            'db_val': sorted(db_val_pids),
            'q_test': sorted(q_test_pids),
            'db_test': sorted(db_test_pids)}
        if rank == 0:
            write_json(splits, osp.join(self.root, 'splits.json'))
        synchronize()
