from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import (pairwise_distances, precision_recall_curve, 
                             plot_precision_recall_curve, average_precision_score, 
                             PrecisionRecallDisplay)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from .pca import PCA
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils.dist_utils import synchronize
from .utils.serialization import write_json
from .utils.data.preprocessor import Preprocessor
from .utils import to_torch

import os

def extract_cnn_feature(model, inputs, vlad=True, gpu=None):
    model.eval()
    inputs = to_torch(inputs).cuda(gpu)
    outputs = model(inputs)
    if (isinstance(outputs, list) or isinstance(outputs, tuple) and len(outputs)!=3):
        x_pool, x_vlad = outputs
        if vlad:
            outputs = F.normalize(x_vlad, p=2, dim=-1)
        else:
            outputs = F.normalize(x_pool, p=2, dim=-1)
    # for autoencoder
    elif(isinstance(outputs, tuple) and len(outputs)==3):
        features , encoded, _ = outputs
        if vlad:
            outputs = F.normalize(encoded, p=2, dim=-1)
        else:
            N,_,_,_ = features.size()
            features = features.view(N, -1)
            outputs = F.normalize(features, p=2, dim=-1)
    else:
        outputs = F.normalize(outputs, p=2, dim=-1)
    # print(outputs.size())
    return outputs

def extract_infeature(model, inputs, gpu=None):
    model.eval()
    inputs = to_torch(inputs).cuda(gpu)
    outputs = model(inputs)
    if (isinstance(outputs, list) or isinstance(outputs, tuple) and len(outputs)!=3):
        inFeatures, _ = outputs
    return inFeatures
    
def extract_features(model, data_loader, dataset, print_freq=10,
                vlad=True, pca=None, gpu=None, sync_gather=False, return_layer="conv5"):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    features = []

    if (pca is not None):
        pca.load(gpu=gpu)

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, _, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs, vlad, gpu=gpu)
            # 可视化特征图
            # if isvisualized == True:
                # infeatures = extract_infeature(model, imgs, gpu=gpu)
                # visualize(infeatures, fnames, return_layer, visType, imgs.shape[-2], imgs.shape[-1])
            if (pca is not None):
                outputs = pca.infer(outputs)
            outputs = outputs.data.cpu()

            features.append(outputs)

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0 and rank==0):
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    if (pca is not None):
        del pca

    if (sync_gather):
        # all gather features in parallel
        # cost more GPU memory but less time
        features = torch.cat(features).cuda(gpu)
        all_features = [torch.empty_like(features) for _ in range(world_size)]
        dist.all_gather(all_features, features)
        del features
        all_features = torch.cat(all_features).cpu()[:len(dataset)]
        features_dict = OrderedDict()
        for fname, output in zip(dataset, all_features):
            features_dict[fname[0]] = output
        del all_features
    else:
        # broadcast features in sequence
        # cost more time but less GPU memory
        bc_features = torch.cat(features).cuda(gpu)
        features_dict = OrderedDict()
        for k in range(world_size):
            bc_features.data.copy_(torch.cat(features))
            if (rank==0):
                print("gathering features from rank no.{}".format(k))
            dist.broadcast(bc_features, k)
            l = bc_features.cpu().size(0)
            for fname, output in zip(dataset[k*l:(k+1)*l], bc_features.cpu()):
                features_dict[fname[0]] = output
        del bc_features, features

    return features_dict

def pairwise_distance(features, query=None, gallery=None, metric=None):
    # 返回欧氏距离(x-y)^2, x为query, y为database,输出为(query_dim, db_dim)
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m, None, None

    if (dist.get_rank()==0):
        print ("===> Start calculating pairwise distances")
    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def spatial_nms(pred, db_ids, topN):
    assert(len(pred)==len(db_ids))
    pred_select = pred[:topN]
    pred_pids = [db_ids[i] for i in pred_select]
    # find unique
    seen = set()
    seen_add = seen.add
    pred_pids_unique = [i for i, x in enumerate(pred_pids) if not (x in seen or seen_add(x))]
    return [pred_select[i] for i in pred_pids_unique]

def evaluate_all(distmat, gt, gallery, recall_topk=[1, 5, 10, 20], nms=False):
    sort_idx = np.argsort(distmat, axis=1)  # 沿着列向右(每行)的元素进行排序, 由小到大
    del distmat
    db_ids = [db[1] for db in gallery]

    if (dist.get_rank()==0):
        print("===> Start calculating recalls")
    correct_at_n = np.zeros(len(recall_topk))

    for qIx, pred in enumerate(sort_idx):
        if (nms):
            pred = spatial_nms(pred.tolist(), db_ids, max(recall_topk)*12)

        for i, n in enumerate(recall_topk):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recalls = correct_at_n / len(gt)
    del sort_idx

    if (dist.get_rank()==0):
        print('Recall Scores:')
        for i, k in enumerate(recall_topk):
            print('  top-{:<4}{:12.1%}'.format(k, recalls[i]))
    return recalls

def l2_to_cosSim(l2dis):
    return (2 - l2dis) / 2

def evaluate_pr(distmat, gt, gallery, pr_image_path=None, distri_curve_path=None):
    sort_idx = np.argsort(distmat, axis=1)
    cos_sim = []
    y_labels = []
    # 选择L2距离最小的匹配样本
    for qIx, pred in enumerate(sort_idx[:, 0]):
        if gt[qIx]==[]:
            continue
        cos_sim.append(l2_to_cosSim(distmat[qIx][pred]))
        if pred in gt[qIx]:
            y_labels.append(1)
        else:
            y_labels.append(0)
    precision, recall, _ = precision_recall_curve(y_labels, cos_sim)
    auc = average_precision_score(y_labels, cos_sim)
    if (dist.get_rank()==0):
        print('AUC={}'.format(auc))
    if pr_image_path is not None:
        # 存储precision, recall, auc，方便后续的交叉对比
        store_precision_path = pr_image_path.strip('.png') + "_precision.npy"
        np.save(store_precision_path, precision)
        store_recall_path = pr_image_path.strip('.png') + "_recall.npy"
        np.save(store_recall_path, recall)
        store_auc_path = pr_image_path.strip('.png') + "_auc.npy"
        np.save(store_auc_path, np.array(auc))
        draw_pr_curve(recall, precision, auc, pr_image_path)
    if distri_curve_path is not None:
        trueMatches = []
        falseMatches = []
        for qIx in range(distmat.shape[0]):
            for dbIx in range(distmat.shape[1]):
                if dbIx in gt[qIx]:
                    trueMatches.append(distmat[qIx][dbIx])
                else:
                    falseMatches.append(distmat[qIx][dbIx])
        trueMatches_path = distri_curve_path.strip('.png') + "_trueMatches.npy"
        np.save(trueMatches_path, np.array(trueMatches))
        falseMatches_path = distri_curve_path.strip('.png') + "_falseMatches.npy"
        np.save(falseMatches_path, np.array(falseMatches))
        draw_distribution(trueMatches, falseMatches, distri_curve_path)
    return auc

def storeMatchingExamples(distmat, gt, query, gallery, matching_examples_path=None):
    if matching_examples_path is not None:
        sort_idx = np.argsort(distmat, axis=1)
        matching_examples = open(matching_examples_path, 'w')
        for qIx, pred in enumerate(sort_idx[:, 0]):
            if gt[qIx]==[]:
                continue
            if pred in gt[qIx]:
                matching_examples.write("{}, {}, {}, {}\n".format(os.path.basename(query[qIx][0]), os.path.basename(gallery[pred][0]), str([os.path.basename(gallery[i][0]) for i in gt[qIx]]), "True"))
            else:
                matching_examples.write("{}, {}, {}, {}\n".format(os.path.basename(query[qIx][0]), os.path.basename(gallery[pred][0]), str([os.path.basename(gallery[i][0]) for i in gt[qIx]]), "False"))
        matching_examples.close()

def draw_pr_curve(recall, precision, auc, image_path):
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(color='red', label='AP={}'.format(auc))
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Netvlad')
    plt.ylim((0,1.01))
    plt.xlim((0,1))
    plt.savefig(image_path)
    plt.close()

def draw_distribution(trueMatches, falseMatches, distri_curve_path):
    """评估query-database中, 模型对于正样本和负样本的区分度
    
    Args:
        trueMatches: m个欧氏距离
        falseMatches: n个欧氏距离
        distri_curve_path: 分布图存储位置
    
    Returns:
        存储分布直方图以及它们的均值方差
    """
    plt.figure()
    true_mean = np.mean(trueMatches)
    true_std = np.var(trueMatches)
    false_mean = np.mean(falseMatches)
    false_std = np.var(falseMatches)
    counts_x, bins_x = np.histogram(trueMatches,bins=20)
    plt.hist(bins_x[:-1], bins_x, weights=counts_x/len(trueMatches), color="darkorange", edgecolor = 'black', alpha=0.5, label="True matches(u={:.3f}, std={:.3f})".format(true_mean, true_std))
    counts_y, bins_y = np.histogram(falseMatches,bins=20)
    plt.hist(bins_y[:-1], bins_y, weights=counts_y/len(falseMatches), color="darkcyan", edgecolor = 'black', alpha=0.5, label="False matches(u={:.3f}, std={:.3f})".format(false_mean, false_std))
    plt.legend()
    plt.xlabel('L2 Distance')
    plt.ylabel('Probability')
    plt.title('Netvlad')
    plt.savefig(distri_curve_path)
    plt.close()

# 可视化cnn特征图

def visualize(infeatures, fnames, returnlayer="conv5", visType="max", height=480, width=640):
    # infeature is (B,D,W,H)
    if visType == "max":
        infeatures = torch.abs(infeatures)
        infeatures,_ = torch.max(infeatures, dim=1)
    elif visType == "pow2":
        infeatures = infeatures.pow(2).mean(1)
    elif visType == "abs_sum":
        infeatures = torch.abs(infeatures)
        infeatures = infeatures.mean(1)
    
    # store dir
    store_dir = os.path.join('/'.join(fnames[0].split('/')[:-2]), "visualized")
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    # resize成原图片大小
    infeatures = infeatures.unsqueeze(1)  # (B, 1, H, W)
    infeatures = torch.nn.functional.interpolate(infeatures, (height, width), mode="bilinear", align_corners=False)
    
    for index, infeature in enumerate(infeatures):
        infeature = infeature.squeeze().cpu().numpy()  # (B, H, W)
        # 选取95%位置的升序排列的数字作为最大值
        # vmax = np.percentile(infeature, 95)  
        # 初始化归一化模板
        normalizer = mpl.colors.Normalize(vmin=infeature.min(), vmax=infeature.max())
        # 该方法用于scalar data to RGBA mapping，用于可视化
        mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        colormapped_im = (mapper.to_rgba(infeature)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        # store path
        basename = os.path.basename(fnames[index]).strip('.jpg') + "_{}_{}.jpg".format(returnlayer, visType)
        store_path = os.path.join(store_dir, basename)
        im.save(store_path)

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
        self.rank = dist.get_rank()

    def evaluate(self, query_loader, dataset, query, gallery, ground_truth, gallery_loader=None, \
                    vlad=True, pca=None, rerank=False, gpu=None, sync_gather=False, \
                    nms=False, rr_topk=25, lambda_value=0, pr_image_path=None, distri_curve_path=None,\
                    matching_examples_path=None, return_layer="conv5"):
        if (gallery_loader is not None):
            features = extract_features(self.model, query_loader, query,
                                        vlad=vlad, pca=pca, gpu=gpu, sync_gather=sync_gather, return_layer=return_layer)
            features_db = extract_features(self.model, gallery_loader, gallery,
                                        vlad=vlad, pca=pca, gpu=gpu, sync_gather=sync_gather, return_layer=return_layer)
            features.update(features_db)
        else:
            features = extract_features(self.model, query_loader, dataset,
                            vlad=vlad, pca=pca, gpu=gpu, sync_gather=sync_gather)

        distmat, _, _ = pairwise_distance(features, query, gallery)
        recalls = evaluate_all(distmat, ground_truth, gallery, nms=nms)  # 计算recall
        auc = evaluate_pr(distmat, ground_truth, gallery, pr_image_path=pr_image_path, distri_curve_path=distri_curve_path)  # 绘制pr曲线, 计算auc; 绘制距离分布图
        storeMatchingExamples(distmat, ground_truth, query, gallery, matching_examples_path=matching_examples_path)  # 存储matching file
        
        if (not rerank):
            return auc, recalls

        if (self.rank==0):
            print('Applying re-ranking ...')
            distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
            distmat_qq, _, _ = pairwise_distance(features, query, query)
            distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(),
                                k1=rr_topk, k2=1, lambda_value=lambda_value)

        return evaluate_all(distmat, ground_truth, gallery, nms=nms)
    
        

