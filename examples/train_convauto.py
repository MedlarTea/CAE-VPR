from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import h5py
import scipy.io
import copy

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed as datadist
import torchvision.transforms as T

from ibl import datasets
from ibl import models
from ibl.trainers import VggAutoTrainer
from ibl.evaluators import Evaluator, extract_features, pairwise_distance
from ibl.utils.data import IterLoader, get_transformer_train, get_transformer_test
from ibl.utils.data.sampler import DistributedRandomTupleSampler, DistributedSliceSampler
from ibl.utils.data.preprocessor import Preprocessor
from ibl.utils.logging import Logger
from ibl.pca import PCA
from ibl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from ibl.utils.dist_utils import init_dist, synchronize, convert_sync_bn

start_epoch = best_recall5 = 0

def get_data(args, iters):
    # use robotcar to train vggConvauto
    root = osp.join(args.data_dir, "robotcar")
    images_dir = root
    dataset = datasets.create("robotcar", root, datelist=osp.join(root, 'datelist_for_convAuto.txt'), q_date='2014-12-09-13-21-02', db_val_date='2014-12-10-18-10-50', db_test_date='2014-12-16-18-44-24')

    # use pitts30k to train vggConvauto
    # root = osp.join(args.data_dir, "pitts")
    # images_dir = osp.join(root, "raw")
    # dataset = datasets.create("pitts", root, scale="30k")

    # mean and std for NetVLAD's matconvNet
    train_transformer = get_transformer_train(args.height, args.width)
    test_transformer = get_transformer_test(args.height, args.width)
    # train_transformer = get_transformer_alexnetMatconvnet(args.height, args.width)
    # test_transformer = get_transformer_alexnetMatconvnet(args.height, args.width)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset.q_train+dataset.db_train)  # shuffle default is True

    # train_loader = DataLoader(Preprocessor(dataset.q_train+dataset.db_train, root=images_dir, transform=train_transformer),
    #                         batch_size=args.bs, num_workers=args.workers, sampler=sampler,
    #                         shuffle=False, pin_memory=True)
    train_loader = DataLoader(Preprocessor(dataset.q_train+dataset.db_train, root=root, transform=train_transformer, isRobotcar=True),
                            batch_size=args.bs, num_workers=args.workers, sampler=sampler,
                            shuffle=False, pin_memory=True)

    val_loader = DataLoader(
        Preprocessor(sorted(list(set(dataset.q_val) | set(dataset.db_val))),
                     root=dataset.images_dir, transform=test_transformer, isRobotcar=True),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(sorted(list(set(dataset.q_val) | set(dataset.db_val)))),
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(sorted(list(set(dataset.q_test) | set(dataset.db_test))),
                     root=dataset.images_dir, transform=test_transformer, isRobotcar=True),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(sorted(list(set(dataset.q_test) | set(dataset.db_test)))),
        shuffle=False, pin_memory=True)

    return dataset, train_loader, val_loader, test_loader, sampler

def get_model(args):
    if args.arch == 'vgg':
        base_model = models.create("vgg16", cut_at_pooling=True, matconvnet='logs/vd16_offtheshelf_conv5_3_max.pth', pretrained=True)
        convAuto_model = models.create("convauto", d1=args.d1, d2=args.d2, dimension=args.dimension)
        model = models.create('vggconvauto', base_model, convAuto_model, islayerNorm=args.islayerNorm)

    elif args.arch == 'alexnet':
        base_model = models.create("alexnet", cut_layer="conv5", matconvnet='logs/conv/alexnet/imagenet_matconvnet_alex.pth', isForConvautoTrain=True)
        convAuto_model = models.create("convAutoAlextnet", d1=args.d1, d2=args.d2, dimension=args.dimension)
        model = models.create('alexnetauto', base_model, convAuto_model, islayerNorm=args.islayerNorm)

    if (args.syncbn):
        # not work for VGG16
        convert_sync_bn(model)

    model.cuda(args.gpu)
    model = nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu
            )
    return model

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    global start_epoch, best_recall5
    init_dist(args.launcher, args)
    synchronize()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # if args.deterministic:
    #     cudnn.deterministic = True
    #     cudnn.benchmark = False

    print("Use GPU: {} for training, rank no.{} of world_size {}"
          .format(args.gpu, args.rank, args.world_size))

    if (args.rank==0):
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset, train_loader, val_loader, test_loader, sampler = get_data(args, iters)

    # Create model
    model = get_model(args)

    # Load from vgg_trained checkpoint
    if args.arch == 'vgg' and args.vgg16_resume:
        checkpoint = load_checkpoint(args.vgg16_resume)
        copy_state_dict(checkpoint['state_dict'], model)
        _best_recall5 = checkpoint['best_recall5']
        if (args.rank==0):
            print("=>best recall5 {:.1%}"
                  .format(_best_recall5))

    # Evaluator
    evaluator = Evaluator(model)
    if (args.rank==0):
        print("Test the initial model:")
    auc, recalls = evaluator.evaluate(val_loader, sorted(list(set(dataset.q_val) | set(dataset.db_val))),
                        dataset.q_val, dataset.db_val, dataset.val_pos,
                        vlad=True, gpu=args.gpu, sync_gather=args.sync_gather, isRobotcar = True)

    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  # Adam优化器

    # Trainer
    trainer = VggAutoTrainer(model, gpu=args.gpu)

    # Start training
    for epoch in range(0, args.epochs):
        sampler.set_epoch(args.seed+epoch)

        g = torch.Generator()
        g.manual_seed(args.seed+epoch)

        synchronize()
        trainer.train(epoch, train_loader, optimizer,
                        train_iters=len(train_loader), print_freq=args.print_freq)
        synchronize()


        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            auc, recalls = evaluator.evaluate(val_loader, sorted(list(set(dataset.q_val) | set(dataset.db_val))),
                                    dataset.q_val, dataset.db_val, dataset.val_pos,
                                    vlad=True, gpu=args.gpu, sync_gather=args.sync_gather, isRobotcar = True)

            is_best = recalls[1] > best_recall5
            best_recall5 = max(recalls[1], best_recall5)

            if (args.rank==0):
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_recall5': best_recall5,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint'+str(epoch)+'.pth.tar'))
                print('\n * Finished epoch {:3d} recall@1: {:5.1%}  recall@5: {:5.1%}  recall@10: {:5.1%}  best@5: {:5.1%}{}\n'.
                      format(epoch, recalls[0], recalls[1], recalls[2], best_recall5, ' *' if is_best else ''))

        synchronize()

    # final inference
    if (args.rank==0):
        print("Testing on TestSet:")
    evaluator.evaluate(test_loader, sorted(list(set(dataset.q_test) | set(dataset.db_test))),
                dataset.q_test, dataset.db_test, dataset.test_pos,
                vlad=True, gpu=args.gpu, sync_gather=args.sync_gather, isRobotcar = True)
    synchronize()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NetVLAD/SARE training")
    parser.add_argument('--launcher', type=str,
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--tcp-port', type=str, default='5017')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='pitts',
                        choices=datasets.names())
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help="tuple numbers in a batch")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")
    # model
    parser.add_argument('--arch', type=str, default='vgg', choices=['vgg', 'alexnet'])
    parser.add_argument('--dimension', type=int, default=1024)
    parser.add_argument('--d1', type=int, default=512)
    parser.add_argument('--d2', type=int, default=512)
    parser.add_argument('--layers', type=str, default='conv5')
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--sync-gather', action='store_true')
    parser.add_argument('--islayerNorm', action='store_true')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--step-size', type=int, default=5)
    # training configs
    parser.add_argument('--vgg16_resume', type=str, default='', metavar='PATH')

    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=0)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--print-freq', type=int, default=10)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--init-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '..', 'logs'))
                    
    main()
