import os.path as osp
import pandas as pd
import torch.distributed as dist

from collections import namedtuple
from ..utils.data.dataset import DatasetRobotcar
from ..utils.serialization import write_json
from ..utils.dist_utils import synchronize

def read_csv(date_dir, csv):
    timestamps = []
    utms = []
    degrees = []
    for index, row in csv.iterrows():
        timestamps.append([osp.join(date_dir, "stereo/centre", str(int(row["timestamps"]))+".jpg")])
        utms.append([row["x"], row["y"]])
        degrees.append(row["degree"])
    return timestamps, utms, degrees, len(timestamps)

def parse_dataset(date_dir, pr_csv):
    df = pd.read_csv(pr_csv)
    trainSet = df[df["dataset_type"]==2]
    validSet = df[df["dataset_type"]==3]
    testSet = df[df["dataset_type"]==1]
    trainTimestamps, trainUtms, train_degrees, train_nums = read_csv(date_dir, trainSet)
    validTimestamps, validUtms, valid_degrees, valid_nums = read_csv(date_dir, validSet)
    testTimestamps, testUtms, test_degrees, test_nums = read_csv(date_dir, testSet)
    dbStruct = namedtuple('dbStruct',['trainTimestamps', 'trainUtms', 'train_degrees', 'train_nums', 
                                        'validTimestamps', 'validUtms', 'valid_degrees', 'valid_nums',
                                        'testTimestamps', 'testUtms', 'test_degrees', 'test_nums'])
    return dbStruct(trainTimestamps, trainUtms, train_degrees, train_nums, 
                    validTimestamps, validUtms, valid_degrees, valid_nums,
                    testTimestamps, testUtms, test_degrees, test_nums)

class Robotcar(DatasetRobotcar):
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

    def __init__(self, root, datelist, q_date, db_val_date, db_test_date, verbose=True):
        super(Robotcar, self).__init__(root)
        self.datelist = datelist
        self.q_date = q_date
        self.db_val_date = db_val_date
        self.db_test_date = db_test_date

        self.arrange()
        self.load(verbose)

    def arrange(self):
        # if self._check_integrity():
        #     return

        try:
            rank = dist.get_rank()
        except:
            rank = 0

        # the root path for raw dataset
        raw_dir = osp.join(self.root, 'oxford')
        if (not osp.isdir(raw_dir)):
            raise RuntimeError("Dataset not found.")
        
        identities = []
        utms = []
        degrees = []
        q_train_pids = []
        db_train_pids = []
        q_val_pids = []
        db_val_pids = []
        q_test_pids = []
        db_test_pids = []
        with open(osp.join(self.root, self.datelist)) as file:
            for line in file:
                date_dir = osp.join(raw_dir, line.strip())
                csv_file = osp.join(date_dir, "pr_dataset.csv")
                struct = parse_dataset(date_dir, csv_file)
                if line.strip() == self.q_date:
                    q_train_pids += [i for i in range(len(identities), len(identities)+struct.train_nums)]
                    identities += struct.trainTimestamps
                    utms += struct.trainUtms
                    degrees += struct.train_degrees
                    q_val_pids += [i for i in range(len(identities), len(identities)+struct.valid_nums)]
                    identities += struct.validTimestamps
                    utms += struct.validUtms
                    degrees += struct.valid_degrees
                    q_test_pids += [i for i in range(len(identities), len(identities)+struct.test_nums)]
                    identities += struct.testTimestamps
                    utms += struct.testUtms
                    degrees += struct.test_degrees
                elif line.strip() == self.db_val_date:
                    db_train_pids += [i for i in range(len(identities), len(identities)+struct.train_nums)]
                    identities += struct.trainTimestamps
                    utms += struct.trainUtms
                    degrees += struct.train_degrees
                    db_val_pids += [i for i in range(len(identities), len(identities)+struct.train_nums+struct.valid_nums+struct.test_nums)]
                    identities += struct.trainTimestamps
                    identities += struct.validTimestamps
                    identities += struct.testTimestamps
                    utms += struct.trainUtms
                    degrees += struct.train_degrees
                    utms += struct.validUtms
                    degrees += struct.valid_degrees
                    utms += struct.testUtms
                    degrees += struct.test_degrees
                elif line.strip() == self.db_test_date:
                    db_train_pids += [i for i in range(len(identities), len(identities)+struct.train_nums)]
                    identities += struct.trainTimestamps
                    utms += struct.trainUtms
                    degrees += struct.train_degrees
                    db_test_pids += [i for i in range(len(identities), len(identities)+struct.train_nums+struct.valid_nums+struct.test_nums)]
                    identities += struct.trainTimestamps
                    identities += struct.validTimestamps
                    identities += struct.testTimestamps
                    utms += struct.trainUtms
                    degrees += struct.train_degrees
                    utms += struct.validUtms
                    degrees += struct.valid_degrees
                    utms += struct.testUtms
                    degrees += struct.test_degrees
                else:
                    db_train_pids += [i for i in range(len(identities), len(identities)+struct.train_nums)]
                    identities += struct.trainTimestamps
                    utms += struct.trainUtms
                    degrees += struct.train_degrees
        assert len(identities)==len(utms)
        meta = {'name': 'Robotcar',
            'identities': identities, 'utm': utms, 'degrees': degrees}
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
