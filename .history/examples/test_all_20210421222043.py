from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import h5py

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed as datadist

from ibl import datasets
from ibl import models
from ibl.evaluators import Evaluator, extract_features, pairwise_distance
from ibl.utils.data import IterLoader, get_transformer_train, get_transformer_test
from ibl.utils.data.sampler import DistributedSliceSampler
from ibl.utils.data.preprocessor import Preprocessor
from ibl.utils.logging import Logger
from ibl.pca import PCA
from ibl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict, write_json
from ibl.utils.dist_utils import init_dist, synchronize


def get_data(args):
    allDatasets = {}
    # pitts
    root = osp.join(args.data_dir, "pitts")
    allDatasets["pitts"] = datasets.create("pitts", root, scale="30k")
    # tokyo
    root = osp.join(args.data_dir, "tokyo")
    allDatasets["tokyo"] = datasets.create("tokyo", root)

    # mapillary
    # root = osp.join(args.data_dir, "mapillary")
    # with open(osp.join(root, "citys.txt")) as f:
    #     for line in f:
    #         allDatasets["mapillary_"+line.strip().split('/')[-1]] = datasets.create("mapillary", root, city=line.strip(), isUseNight=True)

    # robotcar, q_date vs. db_test_date
    root = osp.join(args.data_dir, "robotcar")
    allDatasets["robotcar_qAutumn_dbSunCloud"] = datasets.create("robotcar", root, datelist=osp.join(root, 'datelist_for_convAuto.txt'), q_date='2014-12-09-13-21-02', db_val_date='2014-12-16-18-44-24', db_test_date='2014-11-18-13-20-12')
    allDatasets["robotcar_qAutumn_dbNight"] = datasets.create("robotcar", root, datelist=osp.join(root, 'datelist_for_convAuto.txt'), q_date='2014-12-09-13-21-02', db_val_date='2014-11-18-13-20-12', db_test_date='2014-12-16-18-44-24')
    allDatasets["robotcar_qSnow_dbNight"] = datasets.create("robotcar", root, datelist=osp.join(root, 'datelist_for_convAuto.txt'), q_date='2015-02-03-08-45-10', db_val_date='2014-11-18-13-20-12', db_test_date='2014-12-16-18-44-24')
    allDatasets["robotcar_qSnow_dbSunCloud"] = datasets.create("robotcar", root, datelist=osp.join(root, 'datelist_for_convAuto.txt'), q_date='2015-02-03-08-45-10', db_val_date='2014-12-16-18-44-24', db_test_date='2014-11-18-13-20-12')

    # uacampus
    root = osp.join(args.data_dir, "uacampus")
    allDatasets["uacampus"] = datasets.create("uacampus", root)

    

    # for single frame
    root = osp.join(args.data_dir, "singleframe")
    allDatasets["BerlinA100_RAS2020"] = datasets.create("singleframe", root, city="BerlinA100_RAS2020")
    allDatasets["Halen_RAS2020"] = datasets.create("singleframe", root, city="Halen_RAS2020")
    allDatasets["Kudamm_RAS2020"] = datasets.create("singleframe", root, city="Kudamm_RAS2020", isjpg=True)
    allDatasets["Gardens_point_RAS2020"] = datasets.create("singleframe", root, city="Gardens_point_RAS2020", isjpg=True)
    allDatasets["Nordland_RAS2020"] = datasets.create("singleframe", root, city="Nordland_RAS2020")


    test_transformer_db = get_transformer_test(args.height, args.width)
    test_transformer_q = get_transformer_test(args.height, args.width, tokyo=(args.dataset=='tokyo'))

    # for PCA
    # dataset = datasets.create('pitts', osp.join(args.data_dir, 'pitts'), scale='30k', verbose=False)
    root = osp.join(args.data_dir, "robotcar")
    dataset = datasets.create("robotcar", root, datelist=osp.join(root, 'datelist_PR_Attention.txt'), q_date='2015-02-03-08-45-10', db_val_date='2015-11-10-14-15-57', db_test_date='2015-07-10-10-01-59') 

    pitts_train = sorted(list(set(dataset.q_train) | set(dataset.db_train)))
    train_extract_loader = DataLoader(
        Preprocessor(pitts_train, root=dataset.images_dir, transform=test_transformer_db, isRobotcar=True),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(pitts_train),
        shuffle=False, pin_memory=True)

    test_loader_qs = {}
    test_loader_dbs = {}
    for name in allDatasets.keys():
        test_loader_q = DataLoader(
            Preprocessor(allDatasets[name].q_test, root=allDatasets[name].images_dir, transform=test_transformer_q, isRobotcar=True if name.find("robotcar")!=-1 else False),
            batch_size=(1 if args.dataset=='tokyo' else args.test_batch_size), num_workers=args.workers,
            sampler=DistributedSliceSampler(allDatasets[name].q_test),
            shuffle=False, pin_memory=True)
        test_loader_qs[name] = test_loader_q

        test_loader_db = DataLoader(
            Preprocessor(allDatasets[name].db_test, root=allDatasets[name].images_dir, transform=test_transformer_db, isRobotcar=True if name.find("robotcar")!=-1 else False),
            batch_size=args.test_batch_size, num_workers=args.workers,
            sampler=DistributedSliceSampler(allDatasets[name].db_test),
            shuffle=False, pin_memory=True)
        test_loader_dbs[name] = test_loader_db
        

    return allDatasets, pitts_train, train_extract_loader, test_loader_qs, test_loader_dbs

def get_model(args):
    # for netvlad
    base_model = models.create("vgg16", cut_at_pooling=False, matconvnet='logs/vd16_offtheshelf_conv5_3_max.pth', pretrained=True, return_layer=args.return_layer)
    if args.vlad:
        pool_layer = models.create('netvlad', dim=base_model.feature_dim)
        model = models.create('embednet', base_model, pool_layer)
    else:
        model = base_model



    # # for vgg16+ConvAuto, note:args.vlad==true
    # if args.arch == 'vgg16':
    #     base_model = models.create("vgg16", cut_at_pooling=True, matconvnet='logs/vd16_offtheshelf_conv5_3_max.pth', pretrained=True)
    #     convAuto_model = models.create("convauto", dimension=args.dimension)
    #     model = models.create('vggconvauto', base_model, convAuto_model, islayerNorm=args.islayerNorm)
    # # for alexnet + ConAuto, note:args.vlad==true
    # elif args.arch == 'alexnet':
    #     base_model = models.create("alexnet", cut_layer="conv5", matconvnet='logs/imagenet_matconvnet_alex.pth', isForConvautoTrain=True)
    #     convAuto_model = models.create("convAutoAlextnet", dimension=args.dimension)
    #     model = models.create('alexnetauto', base_model, convAuto_model, islayerNorm=args.islayerNorm)

    # for vgg16, note:args.vlad==false
    # base_model = models.create("vgg16", cut_at_pooling=True, matconvnet='logs/vd16_offtheshelf_conv5_3_max.pth', pretrained=True)
    # convAuto_model = models.create("convauto", dimension=args.dimension)
    # model = models.create('vggconvauto', base_model, convAuto_model, islayerNorm=args.islayerNorm)
    # for alexnet
    # model = models.create("alexnet", cut_layer="conv5", matconvnet='logs/alexnet/imagenet_matconvnet_alex.pth')

    

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
    init_dist(args.launcher, args)
    synchronize()
    cudnn.benchmark = True
    print("Use GPU: {} for testing, rank no.{} of world_size {}"
          .format(args.gpu, args.rank, args.world_size))

    assert(args.resume)
    if(args.rank==0):
        print(args)

    # Create data loaders
    allDatasets, pitts_train, train_extract_loader, test_loader_qs, test_loader_dbs = get_data(args)

    # Create model
    model = get_model(args)
    # print(model)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_recall5 = checkpoint['best_recall5']
        if (args.rank==0):
            print("=> Start epoch {}  best recall5 {:.1%}"
                  .format(start_epoch, best_recall5))

    # Evaluator
    evaluator = Evaluator(model)
    if (args.reduction):
        pca_parameters_path = osp.join(osp.dirname(args.resume), 'pca_params_'+osp.basename(args.resume).split('.')[0]+'.h5')
        pca = PCA(args.features, (not args.nowhiten), pca_parameters_path)
        if (not osp.isfile(pca_parameters_path)):
            dict_f = extract_features(model, train_extract_loader, pitts_train,
                    vlad=args.vlad, gpu=args.gpu, sync_gather=args.sync_gather)
            features = list(dict_f.values())
            if (len(features)>10000):
                features = random.sample(features, 10000)
            features = torch.stack(features)
            if (args.rank==0):
                pca.train(features)
            synchronize()
            del features
    else:
        pca = None
    
    # 记录结果
    if (args.rank==0):
        log_dir = osp.dirname(args.resume)
        result_path = osp.join(log_dir, "log_test.txt")
        pResult = open(result_path, "w")
    # for pr-curve
    for name in allDatasets.keys():
        if (len(allDatasets[name].q_test) > 8000):
            continue
        if (args.rank==0):
            print("==========\nDataset:{}\n==========".format(name))
            pResult.write("==========Dataset:{}==========\n".format(name))
            print("  test_query    | {:8d}"
                  .format(len(allDatasets[name].q_test)))
            pResult.write("  test_query    | {:8d}\n".format(len(allDatasets[name].q_test)))
            print("  test_gallery  | {:8d}"
                  .format(len(allDatasets[name].db_test)))
            pResult.write("  test_gallery  | {:8d}\n".format(len(allDatasets[name].db_test)))
            

        pr_curve_path = osp.join(osp.dirname(args.resume), name+"_pr.png")
        distri_curve_path = osp.join(osp.dirname(args.resume), name+"_distri.png")
        matching_examples_path = osp.join(osp.dirname(args.resume), name+"_matchingSituation.txt")
        auc, recalls = evaluator.evaluate(test_loader_qs[name], sorted(list(set(allDatasets[name].q_test) | set(allDatasets[name].db_test))),
                            allDatasets[name].q_test, allDatasets[name].db_test, allDatasets[name].test_pos, gallery_loader=test_loader_dbs[name],
                            vlad=args.vlad, pca=pca, rerank=args.rerank, gpu=args.gpu, sync_gather=args.sync_gather,
                            nms=(True if args.dataset=='tokyo' else False),
                            rr_topk=args.rr_topk, lambda_value=args.lambda_value, pr_image_path=pr_curve_path,
                            distri_curve_path=distri_curve_path, matching_examples_path=matching_examples_path, return_layer=args.return_layer, isRobotcar = True if name.find("robotcar")!=-1 else False)
        if (args.rank==0):
            pResult.write("  Auc: %0.3f\n"%auc)
            pResult.write("  Recall@1: %0.3f\n"%recalls[0])
            pResult.write("  Recall@5: %0.3f\n"%recalls[1])
            pResult.write("  Recall@10: %0.3f\n"%recalls[2])
            pResult.write("  Recall@20: %0.3f\n"%recalls[3])
    if (args.rank==0):
        pResult.close()
        # sys.stdout.__exit__()
    synchronize()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-based localization testing")
    parser.add_argument('--launcher', type=str,
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--tcp-port', type=str, default='5017')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='pitts',
                        choices=datasets.names())
    parser.add_argument('--scale', type=str, default='30k')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help="tuple numbers in a batch")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")
    parser.add_argument('--num-clusters', type=int, default=64)
    # model
    parser.add_argument('-a', '--arch', type=str, default='vgg16',
                        choices=models.names())
    parser.add_argument('--nowhiten', action='store_true')
    parser.add_argument('--sync-gather', action='store_true')
    parser.add_argument('--features', type=int, default=4096)
    parser.add_argument('--dimension', type=int, default=4096)
    parser.add_argument('--islayerNorm', type=bool, default=True)
    parser.add_argument('--syncbn', action='store_true')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--vlad', action='store_true')
    parser.add_argument('--reduction', action='store_true',
                        help="evaluation only")
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--rr-topk', type=int, default=25)
    parser.add_argument('--lambda-value', type=float, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    # visualize
    parser.add_argument('--isvisualized', action="store_true", default=False, help= "is visualised")
    parser.add_argument('--visType', type=str, default="pow2", choices=['max', 'pow2', 'abs_sum'],)
    parser.add_argument('--return_layer', type=str, default="conv5", choices=[None, 'conv6', 'conv5', 'conv4', "conv3", "conv2"])
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default="/home/jing/Data/Dataset")
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    main()
