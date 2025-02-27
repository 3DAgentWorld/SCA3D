#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   emd_match_pointnet.py
@Time    :   2021/11/29 19:08:25
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   None
'''

from collections import OrderedDict
import torch
import torchtext
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import functional as F
from SCA3D.models.model_utils.emd_util import emd_inference_qpth, emd_inference_opencv_test
from torch.nn.utils.rnn import pad_sequence


def lambda_softmax_func(inputs, lambda_val, k_parts):
    """[summary]

    Args:
        inputs : =>(batch_size, max_word_len, k_parts) 
    Return:
        =>(batch_size, max_word_len, k_parts)
    """
    inputs = inputs * lambda_val
    # return_val = torch.zeros_like(inputs)
    exp_attn = torch.exp(inputs)
    mask = torch.arange(inputs.shape[2]).view(
        1, 1, -1).cuda().expand_as(inputs)
    k_expand = torch.Tensor(k_parts).view(-1, 1, 1).cuda().expand_as(inputs)
    mask = mask >= k_expand
    exp_attn = exp_attn.masked_fill(mask, 0.)
    exp_sum = exp_attn.sum(dim=2, keepdim=True)
    return_val = exp_attn / exp_sum
    return return_val


def get_emd_score(im, s, s_l, k_part, opt):
    # cal the similarity matrix
    bs_i = im.size(0)
    bs_s = s.size(0)
    n_region = im.size(1)
    n_word = s.size(1)
    dim = im.size(-1)
    # im -> (bs, n_parts, d)
    # s -> (bs, n_words, d)
    dot_p = torch.bmm(im.reshape(1, -1, dim), s.reshape(1, -1, dim).transpose(1, 2))
    dot_p = dot_p.reshape(bs_i, n_region, bs_s, n_word).transpose(1, 2).reshape(bs_i * bs_s, n_region, n_word)
    # dot_p -> (bs*bs, n_parts, n_words)

    tri_part = torch.tril((1. / torch.arange(n_region + 1)).view(-1, 1).repeat(1, n_region), diagonal=-1).cuda()
    w_im = tri_part[k_part].unsqueeze(1).repeat(1, bs_s, 1).reshape(bs_i * bs_s, -1)
    tri_word = torch.tril((1. / torch.arange(n_word + 1)).view(-1, 1).repeat(1, n_word), diagonal=-1).cuda()
    w_txt = tri_word[s_l].unsqueeze(0).repeat(bs_i, 1, 1).reshape(bs_i * bs_s, -1)

    # 计算两两pair的EMD score
    # emd_score, _ = emd_inference_qpth(1 - dot_p, w_im, w_txt) # attn
    cost_mat = 1 - F.relu(dot_p)
    emd_scores, _ = emd_inference_opencv_test(cost_mat, w_im, w_txt)  # attn
    emd_scores = -1 * emd_scores.reshape(bs_i, bs_s)
    return emd_scores


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'rgb':
        img_enc = EncoderRGB(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError(
            "Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderRGB(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderRGB, self).__init__()
        self.embed_size = embed_size
        self.img_dim = img_dim
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(self.img_dim + 256, embed_size)
        self.fc_rgb1 = nn.Linear(3, 128)
        self.fc_rgb2 = nn.Linear(128, 256)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.fc_rgb1.weight)
        torch.nn.init.xavier_uniform_(self.fc_rgb2.weight)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        shape_feature, rgb_feature = torch.split(images, [self.img_dim, 3], -1)
        rgb_feature = F.relu(self.fc_rgb1(rgb_feature))
        rgb_feature = F.relu(self.fc_rgb2(rgb_feature))

        features = torch.cat((shape_feature, rgb_feature), -1)
        features = self.fc(features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderRGB, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        # Embedding init
        self.embed.weight.data.uniform_(-0.1, 0.1)
        torch.nn.init.kaiming_normal_(self.embed.weight)

        # GRU init
        torch.nn.init.orthogonal_(self.rnn.weight_ih_l0)
        torch.nn.init.orthogonal_(self.rnn.weight_hh_l0)
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.bias_hh_l0.data.zero_()

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            slice_bound = int(cap_emb.size(2) / 2)
            cap_emb = (cap_emb[:, :, :slice_bound] +
                       cap_emb[:, :, slice_bound:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))  # .squeeze()


class EMDLoss(nn.Module):
    def __init__(self, opt, HN=True):
        super(EMDLoss, self).__init__()
        self.opt = opt
        self.loss = HardNegativeNCE()

    def forward(self, im, s, s_l, k_part):
        '''
            Note: norm im, s before this function
        '''
        # calculate the EMD loss
        bs = im.size(0)
        n_region = im.size(1)
        n_word = s.size(1)
        dim = im.size(-1)
        # cal the similarity matrix
        # im -> (bs, n_parts, d)
        # s -> (bs, n_words, d)
        dot_p = torch.bmm(im.reshape(1, -1, dim), s.reshape(1, -1, dim).transpose(1, 2))
        dot_p = dot_p.reshape(bs, n_region, bs, n_word).transpose(1, 2).reshape(bs * bs, n_region, n_word)
        # dot_p -> (bs*bs, n_parts, n_words)

        tri_part = torch.tril((1. / torch.arange(n_region + 1)).view(-1, 1).repeat(1, n_region), diagonal=-1).cuda()
        w_im = tri_part[k_part].unsqueeze(1).repeat(1, bs, 1).reshape(bs * bs, -1)
        tri_word = torch.tril((1. / torch.arange(n_word + 1)).view(-1, 1).repeat(1, n_word), diagonal=-1).cuda()
        w_txt = tri_word[s_l].unsqueeze(0).repeat(bs, 1, 1).reshape(bs * bs, -1)

        cost_mat = 1 - F.relu(dot_p)
        _, emd_flow = emd_inference_opencv_test(cost_mat, w_im, w_txt)  # attn
        # emd_score -> (bs*bs)
        # emd_flow -> (bs*bs, n_parts, n_words)
        emd_score = torch.sum(cost_mat * emd_flow, dim=(1, 2))
        emd_score = -1 * emd_score.reshape(bs, bs)
        # emd_score -> (bs, bs)

        loss = self.loss(emd_score)
        return loss


class HardNegativeNCE(nn.Module):
    """
    Hard-Negative NCE loss for contrastive learning.
    https://arxiv.org/pdf/2301.02280.pdf
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.25, temp=0.07, **kwargs):
        """
        Args:
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(HardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temp = temp
        self.eps = 1e-7

    def forward(
        self,
        sim_matrix,
    ):
        """
        Args:
            sim_matrix: (batch_size, batch_size)
        """
        batch_size = sim_matrix.size(0)

        # scale the similarity matrix with the temperature
        sim_matrix = sim_matrix / (self.temp + self.eps)
        sim_matrix = sim_matrix.float()

        nominator = torch.diagonal(sim_matrix)

        beta_sim = self.beta * sim_matrix
        w_v2t = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim)) + self.eps)
        )
        w_t2v = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim)) + self.eps)
        )
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1) + self.eps)
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0) + self.eps)

        hn_nce_loss = (denominator_v2t - nominator).mean() + (
            denominator_t2v - nominator
        ).mean()
        return hn_nce_loss


# semantic segmentation model
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.Tensor(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1,
                                                                                                             9).repeat(
            batchsize, 1)
        # iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=torch.float32).flatten().view(1, -1).repeat(batchsize, 1)
        # iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, inp_size, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        # self.stn = STN3d()
        self.stn = STNkd(k=inp_size)
        self.conv1 = torch.nn.Conv1d(inp_size, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, inp_size, k=2, shape_embed=512, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.inp_size = inp_size
        self.k = k
        self.shape_embed = shape_embed
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(self.inp_size, global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)

        # 512 -> 256
        # 256 -> 256
        # 128 -> 256
        # add(256s) -> 512
        self.conv1_add = torch.nn.Conv1d(512, 256, 1)
        self.conv2_add = torch.nn.Conv1d(256, 256, 1)
        self.conv3_add = torch.nn.Conv1d(128, 256, 1)
        self.conv_add = torch.nn.Conv1d(256, self.shape_embed, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn1_add = nn.BatchNorm1d(256)
        self.bn2_add = nn.BatchNorm1d(256)
        self.bn3_add = nn.BatchNorm1d(256)
        self.bn_add = nn.BatchNorm1d(self.shape_embed)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x_emb1 = F.relu(self.bn1_add(self.conv1_add(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x_emb2 = F.relu(self.bn2_add(self.conv2_add(x)))

        x = F.relu(self.bn3(self.conv3(x)))
        x_emb3 = F.relu(self.bn3_add(self.conv3_add(x)))

        x_embed = x_emb1.add(x_emb2).add(x_emb3)
        x_embed = F.relu(self.bn_add(self.conv_add(x_embed)))

        x_sem = self.conv4(x)

        x_sem = x_sem.transpose(2, 1).contiguous()
        x_sem = F.log_softmax(x_sem.view(-1, self.k), dim=-1)
        x_sem = x_sem.view(batchsize, n_pts, self.k)

        x_embed = x_embed.transpose(2, 1).contiguous()
        x_embed = x_embed.view(batchsize, n_pts, self.shape_embed)
        return x_sem, x_embed, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class SCA3D(object):
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.max_k_part = opt.K
        self.min_point_rate = opt.min_point_rate
        self.stage_1_epoch = opt.stage_1_epoch
        self.alpha = opt.alpha
        self.precomp_enc_type = opt.precomp_enc_type
        self.SEG_NUM = opt.SEG_NUM

        self.pointnet = PointNetDenseCls(opt.inp_size, opt.SEG_NUM, opt.img_dim, feature_transform=True)
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)

        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)

        if torch.cuda.is_available():
            self.pointnet.cuda()
            self.img_enc.cuda()
            self.txt_enc.cuda()

        # Loss and Optimizer
        if opt.matching_method == 'emd':
            self.criterion = EMDLoss(opt=opt)

        # params
        params = list(self.txt_enc.parameters())
        params += list(self.pointnet.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    @staticmethod
    def mask_emb(embedding, n_len):
        """[summary]

        Args:
            embedding: tensor(bs, max_n_parts, d)
            n_len list: list(bs) 
        """
        bs = embedding.size(0)
        max_n_parts = embedding.size(1)
        tri_slice = torch.triu(torch.ones(max_n_parts + 1, max_n_parts, dtype=torch.bool)).cuda()
        mask = tri_slice[n_len]  # (bs, max_n_parts)
        return torch.masked_fill(embedding, mask.unsqueeze(-1), 0)

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.pointnet.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.pointnet.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.pointnet.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.pointnet.eval()

    def forward_emb(self, images, captions, lengths, sentences):
        """Compute the image and caption embeddings
        """

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_loss(self, img_emb, cap_emb, cap_len, k_part, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len, k_part)
        self.logger.update('Matching Loss', loss.item(), img_emb.size(0))
        return loss

    def forward_pointnet(self, points):
        points = points.transpose(2, 1)
        pred, embed, trans, trans_feat = self.pointnet(points)
        return pred, embed, trans_feat

    def forward_semantic_loss(self, pred, target, trans_feat):
        pred = pred.view(-1, self.SEG_NUM)
        target = target.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred, target)
        loss += feature_transform_regularizer(trans_feat) * 0.001
        self.logger.update('Semantic Loss', loss.item(), pred.size(0))

        return loss

    def point2part(self, point_embed, seg_idx):
        '''
            seg_idx => (bs, pts)
        '''
        seg_idx.detach_()
        bs = seg_idx.size()[0]
        pts = seg_idx.size()[1]
        embed_size = point_embed.size()[-1]
        M = torch.zeros(bs, self.SEG_NUM, pts).cuda()
        M.scatter_(1, seg_idx.unsqueeze(1), 1)
        M_mean = F.normalize(M, p=1, dim=-1)
        all_part_embed = torch.bmm(M_mean, point_embed)  # bs, SEG_NUM, point_embed
        part_embed = torch.zeros(bs, self.max_k_part, embed_size).cuda()
        # batched
        # 1. batched count
        seg_cnt = torch.sum(M, dim=2)  # bs, SEG_NUM
        # 2. batched topk
        kth_cnt = torch.kthvalue(seg_cnt, self.SEG_NUM + 1 - self.max_k_part, dim=1, keepdim=True).values
        # 3. batched filter
        mask_tensor = torch.logical_and(seg_cnt >= pts * self.min_point_rate, seg_cnt > kth_cnt)
        k_part = mask_tensor.sum(dim=1).tolist()
        # 4. batched index select
        # 5. merge to part_embed, k_part
        packed_part_embed = torch.masked_select(all_part_embed, mask_tensor.unsqueeze(-1)).reshape(-1, embed_size)
        part_embed = pad_sequence(packed_part_embed.split(k_part, dim=0), batch_first=True)
        return part_embed, k_part

    def train_emb(self, train_data, epoch):
        """One training step given shapes and captions.
        """
        # torch.cuda.empty_cache()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        self.optimizer.zero_grad()
        shapes = train_data["shapes"].cuda()  # cuda float Tensor
        captions = train_data["captions"].cuda()  # cuda Long Tensor
        sentences = train_data["sentences"]
        semantic_labels = train_data["semantic_labels"].cuda()  # cuda Lone Tensor
        lengths = train_data["lengths"]  # List

        sem_pred, point_embed, trans_feat = self.forward_pointnet(shapes)

        loss = self.forward_semantic_loss(sem_pred, semantic_labels, trans_feat)

        if epoch >= self.stage_1_epoch:
            # point_embed (point embed to part embed)
            if self.precomp_enc_type == 'rgb':
                point_embed = torch.cat((point_embed, shapes[:, :, -3:]), dim=-1)

            part_emb, k_part = self.point2part(point_embed, semantic_labels)

            part_emb, cap_emb, cap_lens = self.forward_emb(part_emb, captions, lengths, sentences)
            part_emb = self.mask_emb(part_emb, k_part)

            loss += self.alpha * self.forward_loss(part_emb, cap_emb, cap_lens, k_part)
        else:
            del point_embed
        self.logger.update('Total Loss', loss.item(), shapes.size(0))
        # compute gradient and do SGD step
        assert not torch.any(torch.isnan(loss))
        loss.backward()
        if self.grad_clip > 0 and epoch >= self.stage_1_epoch:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
