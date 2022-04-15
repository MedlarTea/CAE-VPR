from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from .utils.meters import AverageMeter

class Trainer(object):
    #############################
    # Training module for
    # 1. "NetVLAD: CNN architecture for weakly supervised place recognition" (CVPR'16), loss_type='triplet'
    # 2. "Stochastic Attraction-Repulsion Embedding for Large Scale Localization" (ICCV'19), loss_type='sare_ind' or 'sare_joint'
    #############################
    def __init__(self, model, margin=0.3, gpu=None, temp=0.07):
        super(Trainer, self).__init__()
        self.model = model
        self.gpu = gpu
        self.margin = margin
        self.temp = temp

    def train(self, epoch, sub_id, data_loader, optimizer, train_iters,
                        print_freq=1, vlad=True, loss_type='triplet'):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        data_loader.new_epoch()

        for i in range(train_iters):
            inputs = self._parse_data(data_loader.next())
            data_time.update(time.time() - end)

            loss = self._forward(inputs, vlad, loss_type)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}-{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, sub_id, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))


    def _parse_data(self, inputs):
        imgs = [input[0] for input in inputs]
        imgs = torch.stack(imgs).permute(1,0,2,3,4)
        # imgs_size: batch_size*triplet_size*C*H*W
        return imgs.cuda(self.gpu)

    def _forward(self, inputs, vlad, loss_type):
        # print(inputs.size())
        B, N, C, H, W = inputs.size()
        # print(B, N, C, H, W)
        inputs = inputs.contiguous().view(-1, C, H, W)

        outputs_pool, outputs_vlad = self.model(inputs)
        
        if (not vlad):
            # adopt VLAD layer for feature aggregation
            # print(outputs_pool.size())
            return self._get_loss(outputs_pool, loss_type, B, N)
        else:
            # adopt max pooling for feature aggregation
            # print(outputs_vlad.size())
            return self._get_loss(outputs_vlad, loss_type, B, N)

    def _get_loss(self, outputs, loss_type, B, N):
        outputs = outputs.view(B, N, -1)
        L = outputs.size(-1)

        output_negatives = outputs[:, 2:]
        output_anchors = outputs[:, 0]
        output_positives = outputs[:, 1]

        if (loss_type=='triplet'):
            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_positives = output_positives.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            loss = F.triplet_margin_loss(output_anchors, output_positives, output_negatives,
                                            margin=self.margin, p=2, reduction='mean')

        elif (loss_type=='sare_joint'):
            ### original version: euclidean distance
            dist_pos = ((output_anchors - output_positives)**2).sum(1)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = ((output_anchors - output_negatives)**2).sum(1)
            dist_neg = dist_neg.view(B, -1)

            dist = - torch.cat((dist_pos, dist_neg), 1)
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

            ### new version: dot product
            # dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            # dist_pos = dist_pos.diagonal(0)
            # dist_pos = dist_pos.view(B, 1)
            #
            # output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            # output_negatives = output_negatives.contiguous().view(-1, L)
            # dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            # dist_neg = dist_neg.diagonal(0)
            # dist_neg = dist_neg.view(B, -1)
            #
            # dist = torch.cat((dist_pos, dist_neg), 1)/self.temp
            # dist = F.log_softmax(dist, 1)
            # loss = (- dist[:, 0]).mean()

        elif (loss_type=='sare_ind'):
            ### original version: euclidean distance
            dist_pos = ((output_anchors - output_positives)**2).sum(1)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = ((output_anchors - output_negatives)**2).sum(1)
            dist_neg = dist_neg.view(B, -1)

            dist_neg = dist_neg.unsqueeze(2)
            dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            dist = - torch.cat((dist_pos, dist_neg), 2).view(-1, 2)
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

            ### new version: dot product
            # dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            # dist_pos = dist_pos.diagonal(0)
            # dist_pos = dist_pos.view(B, 1)
            #
            # output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            # output_negatives = output_negatives.contiguous().view(-1, L)
            # dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            # dist_neg = dist_neg.diagonal(0)
            # dist_neg = dist_neg.view(B, -1)
            #
            # dist_neg = dist_neg.unsqueeze(2)
            # dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            # dist = torch.cat((dist_pos, dist_neg), 2).view(-1, 2)/self.temp
            # dist = F.log_softmax(dist, 1)
            # loss = (- dist[:, 0]).mean()

        else:
            assert ("Unknown loss function")

        return loss

class SFRSTrainer(object):
    #############################
    # Training module for
    # "Self-supervising Fine-grained Region Similarities for Large-scale Image Localization"
    #############################
    def __init__(self, model, model_cache, margin=0.3,
                    neg_num=10, gpu=None, temp=[0.07,]):
        super(SFRSTrainer, self).__init__()
        self.model = model
        self.model_cache = model_cache

        self.margin = margin
        self.gpu = gpu
        self.neg_num = neg_num
        self.temp = temp

    def train(self, gen, epoch, sub_id, data_loader, optimizer, train_iters,
                    print_freq=1, lambda_soft=0.5, loss_type='sare_ind'):
        self.model.train()
        self.model_cache.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_hard = AverageMeter()
        losses_soft = AverageMeter()
        end = time.time()

        data_loader.new_epoch()

        for i in range(train_iters):

            inputs_easy, inputs_diff = self._parse_data(data_loader.next())
            data_time.update(time.time() - end)

            loss_hard, loss_soft = self._forward(inputs_easy, inputs_diff, loss_type, gen)
            loss = loss_hard + loss_soft*lambda_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_hard.update(loss_hard.item())
            losses_soft.update(loss_soft.item())

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}-{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_hard {:.3f} ({:.3f})\t'
                      'Loss_soft {:.3f} ({:.3f})'
                      .format(epoch, sub_id, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_hard.val, losses_hard.avg,
                              losses_soft.val, losses_soft.avg))

    def _parse_data(self, inputs):
        imgs = [input[0] for input in inputs]
        imgs = torch.stack(imgs).permute(1,0,2,3,4)
        imgs_easy = imgs[:,:self.neg_num+2]
        imgs_diff = torch.cat((imgs[:,0].unsqueeze(1).contiguous(), imgs[:,self.neg_num+2:]), dim=1)
        return imgs_easy.cuda(self.gpu), imgs_diff.cuda(self.gpu)

    def _forward(self, inputs_easy, inputs_diff, loss_type, gen):
        B, _, C, H, W = inputs_easy.size()
        inputs_easy = inputs_easy.contiguous().view(-1, C, H, W)
        inputs_diff = inputs_diff.contiguous().view(-1, C, H, W)

        sim_easy, vlad_anchors, vlad_pairs = self.model(inputs_easy)
        # vlad_anchors: B*1*9*L
        # vlad_pairs: B*(1+neg_num)*9*L
        with torch.no_grad():
            sim_diff_label, _, _ = self.model_cache(inputs_diff) # B*diff_pos_num*9*9
        sim_diff, _, _ = self.model(inputs_diff)

        if (gen==0):
            loss_hard = self._get_loss(vlad_anchors[:,0,0], vlad_pairs[:,0,0], vlad_pairs[:,1:,0], B, loss_type)
        else:
            loss_hard = 0
            for tri_idx in range(B):
                loss_hard += self._get_hard_loss(vlad_anchors[tri_idx,0,0].contiguous(), vlad_pairs[tri_idx,0,0].contiguous(), \
                                                vlad_pairs[tri_idx,1:], sim_easy[tri_idx,1:,0].contiguous().detach(), loss_type)
            loss_hard /= B

        log_sim_diff = F.log_softmax(sim_diff[:,:,0].contiguous().view(B,-1)/self.temp[0], dim=1)
        loss_soft = (- F.softmax(sim_diff_label[:,:,0].contiguous().view(B,-1)/self.temp[gen], dim=1).detach() * log_sim_diff).mean(0).sum()

        return loss_hard, loss_soft

    def _get_hard_loss(self, anchors, positives, negatives, score_neg, loss_type):
        # select the most difficult regions for negatives
        score_arg = score_neg.view(self.neg_num,-1).argmax(1)
        score_arg = score_arg.unsqueeze(-1).unsqueeze(-1).expand_as(negatives).contiguous()
        select_negatives = torch.gather(negatives,1,score_arg)
        select_negatives = select_negatives[:,0]

        return self._get_loss(anchors.unsqueeze(0).contiguous(), \
                            positives.unsqueeze(0).contiguous(), \
                            select_negatives.unsqueeze(0).contiguous(), 1, loss_type)

    def _get_loss(self, output_anchors, output_positives, output_negatives, B, loss_type):
        L = output_anchors.size(-1)

        if (loss_type=='triplet'):
            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_positives = output_positives.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            loss = F.triplet_margin_loss(output_anchors, output_positives, output_negatives,
                                            margin=self.margin, p=2, reduction='mean')

        elif (loss_type=='sare_joint'):
            dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            dist_pos = dist_pos.diagonal(0)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            dist_neg = dist_neg.diagonal(0)
            dist_neg = dist_neg.view(B, -1)

            # joint optimize
            dist = torch.cat((dist_pos, dist_neg), 1)/self.temp[0]
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        elif (loss_type=='sare_ind'):
            dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            dist_pos = dist_pos.diagonal(0)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            dist_neg = dist_neg.diagonal(0)

            dist_neg = dist_neg.view(B, -1)

            # indivial optimize
            dist_neg = dist_neg.unsqueeze(2)
            dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            dist = torch.cat((dist_pos, dist_neg), 2).view(-1, 2)/self.temp[0]
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        else:
            assert ("Unknown loss function")

        return loss

class VggAutoTrainer(object):
    def __init__(self, model, gpu=None):
        super(VggAutoTrainer, self).__init__()
        self.model = model
        self.gpu = gpu
        self.criterion = nn.MSELoss()  # 均方根误差函数
    
    def train(self, epoch, data_loader, optimizer, train_iters, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):

            inputs = self._parse_data(inputs)
            data_time.update(time.time() - end)

            loss = self._forward(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
    def _parse_data(self, inputs):
        imgs = inputs[0]
        return imgs.cuda(self.gpu)

    def _forward(self, inputs):
        features, _, decoded = self.model(inputs)
        loss = self.criterion(decoded, features)
        return loss

class AttentionTrainer(object):
    #############################
    # Training module for
    # Attention
    #############################
    def __init__(self, model, margin=0.3, gpu=None, useSemantics=False, temp=0.07, lambda_vpr=1.0, lambda_att=1.0):
        super(AttentionTrainer, self).__init__()
        self.model = model
        self.gpu = gpu
        self.useSemantics = useSemantics
        self.margin = margin
        self.temp = temp
        self.lambda_vpr = lambda_vpr
        self.lambda_att = lambda_att

    def train(self, epoch, sub_id, data_loader, optimizer, train_iters,
                        print_freq=1, vlad=True, vpr_loss_type='triplet', att_loss_type='mse'):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        vpr_losses = AverageMeter()
        att_losses = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        data_loader.new_epoch()

        for i in range(train_iters):
            inputs = self._parse_data(data_loader.next())
            data_time.update(time.time() - end)

            if self.useSemantics:
                vpr_loss, att_loss, loss = self._forward(inputs, vlad, vpr_loss_type, att_loss_type) 
                vpr_losses.update(vpr_loss.item())
                att_losses.update(att_loss.item())
                losses.update(loss.item())
            else:
                loss = self._forward(inputs, vlad, vpr_loss_type, att_loss_type)
                losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                if self.useSemantics:
                    print('Epoch: [{}-{}][{}/{}]\t'
                        'Time {:.3f} ({:.3f})\t'
                        'Data {:.3f} ({:.3f})\t'
                        'vprLoss {:.3f} ({:.3f})\t'
                        'attLoss {:.3f} ({:.3f})\t'
                        'Loss {:.3f} ({:.3f})'
                        .format(epoch, sub_id, i + 1, train_iters,
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg,
                                vpr_losses.val, vpr_losses.avg,
                                att_losses.val, att_losses.avg,
                                losses.val, losses.avg))
                else:
                    print('Epoch: [{}-{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})'
                    .format(epoch, sub_id, i + 1, train_iters,
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg))


    def _parse_data(self, inputs):
        imgs = [input[0] for input in inputs]
        imgs = torch.stack(imgs).permute(1,0,2,3,4)
        # imgs_size: batch_size*triplet_size*C*H*W
        return imgs.cuda(self.gpu)

    def _forward(self, inputs, vlad, vpr_loss_type, att_loss_type):
        B, N, C, H, W = inputs.size()
        if self.useSemantics:
            rgb = inputs[:,:,:3,:,:].contiguous().view(-1, 3, H, W)  # extract rgb
            semantic = inputs[:,:,3,:,:].contiguous().view(-1, H, W)  # semantic mask--0/1 mask with (B, H, W)
        else:
            rgb = inputs.contiguous().view(-1, C, H, W)

        if vlad:
            att_mask, outputs = self.model(rgb)
        else:
            outputs, att_mask = self.model(rgb)

        vpr_loss = self._get_vpr_loss(outputs, vpr_loss_type, B, N)
        if self.useSemantics:
            att_loss = self._get_att_loss(att_mask, semantic, att_loss_type)
            return self.lambda_vpr*vpr_loss, self.lambda_att*att_loss, self.lambda_vpr*vpr_loss + self.lambda_att*att_loss
        else:
            return vpr_loss

    def _get_vpr_loss(self, outputs, vpr_loss_type, B, N):
        outputs = outputs.view(B, N, -1)
        L = outputs.size(-1)

        output_negatives = outputs[:, 2:]
        output_anchors = outputs[:, 0]
        output_positives = outputs[:, 1]

        if (vpr_loss_type=='triplet'):
            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_positives = output_positives.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            loss = F.triplet_margin_loss(output_anchors, output_positives, output_negatives,
                                            margin=self.margin, p=2, reduction='mean')

        elif (vpr_loss_type=='sare_joint'):
            ### original version: euclidean distance
            dist_pos = ((output_anchors - output_positives)**2).sum(1)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = ((output_anchors - output_negatives)**2).sum(1)
            dist_neg = dist_neg.view(B, -1)

            dist = - torch.cat((dist_pos, dist_neg), 1)
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

            ### new version: dot product
            # dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            # dist_pos = dist_pos.diagonal(0)
            # dist_pos = dist_pos.view(B, 1)
            #
            # output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            # output_negatives = output_negatives.contiguous().view(-1, L)
            # dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            # dist_neg = dist_neg.diagonal(0)
            # dist_neg = dist_neg.view(B, -1)
            #
            # dist = torch.cat((dist_pos, dist_neg), 1)/self.temp
            # dist = F.log_softmax(dist, 1)
            # loss = (- dist[:, 0]).mean()

        elif (vpr_loss_type=='sare_ind'):
            ### original version: euclidean distance
            dist_pos = ((output_anchors - output_positives)**2).sum(1)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = ((output_anchors - output_negatives)**2).sum(1)
            dist_neg = dist_neg.view(B, -1)

            dist_neg = dist_neg.unsqueeze(2)
            dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            dist = - torch.cat((dist_pos, dist_neg), 2).view(-1, 2)
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

            ### new version: dot product
            # dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            # dist_pos = dist_pos.diagonal(0)
            # dist_pos = dist_pos.view(B, 1)
            #
            # output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            # output_negatives = output_negatives.contiguous().view(-1, L)
            # dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            # dist_neg = dist_neg.diagonal(0)
            # dist_neg = dist_neg.view(B, -1)
            #
            # dist_neg = dist_neg.unsqueeze(2)
            # dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            # dist = torch.cat((dist_pos, dist_neg), 2).view(-1, 2)/self.temp
            # dist = F.log_softmax(dist, 1)
            # loss = (- dist[:, 0]).mean()

        else:
            assert ("Unknown loss function")

        return loss

    def _get_att_loss(self, att_mask, semantic_mask, att_loss_type='mse'):
        if att_loss_type == 'l1':
            return F.l1_loss(att_mask, semantic_mask)
        elif att_loss_type == 'mse':
            return F.mse_loss(att_mask, semantic_mask)

class DisentangleNetTrainer(object):
    def __init__(self, model, gpu=None):
        super(DisentangleNetTrainer, self).__init__()
        self.model = model
        self.gpu = gpu
        self.criterion = nn.MSELoss()  # 均方根误差函数
    
    def train(self, epoch, data_loader, optimizer, train_iters, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):

            inputs = self._parse_data(inputs)
            data_time.update(time.time() - end)

            loss = self._forward(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
    def _parse_data(self, inputs):
        imgs = inputs[0]
        return imgs.cuda(self.gpu)

    def _forward(self, inputs):
        output = self.model(inputs)
        loss = self.criterion(output[:,0], output[:,1])
        return loss

class ReconstructTrainer(object):
    def __init__(self, model, gpu=None):
        super(ReconstructTrainer, self).__init__()
        self.model = model
        self.gpu = gpu
        self.criterion = nn.MSELoss()  # 均方根误差函数
    
    def train(self, epoch, data_loader, optimizer, train_iters, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):

            inputs = self._parse_data(inputs)
            data_time.update(time.time() - end)

            loss = self._forward(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
    def _parse_data(self, inputs):
        imgs = inputs[0]
        return imgs.cuda(self.gpu)

    def _forward(self, inputs):
        features = self.model(inputs)
        loss = self.criterion(decoded, features)
        return loss