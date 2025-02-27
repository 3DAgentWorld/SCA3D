#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2020/11/25 20:34:57
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   Data provider
'''

import json
import os
import pickle
import platform
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data

import nltk
from nltk.corpus import wordnet
from string import Template


def add_vocab(i2w, w2i, word):
    idx = len(i2w) + 1
    i2w[idx] = word
    w2i[word] = idx


def rotate_point(point, rotation_angle):
    point = np.array(point)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, 0, -sin_theta],
                                [0, 1, 0],
                                [sin_theta, 0, cos_theta]])
    rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix)
    return rotated_point


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_path, data_split, shapenet_path, vocab, part_num, cfg):
        self.vocab = vocab
        # cfg for point cloud
        self.npoints = cfg.num_points
        self.pkl_path = os.path.join(shapenet_path, cfg.pkl_path)
        self.seg_num = cfg.SEG_NUM

        self.part_num = part_num

        # 读取split
        if data_split == "train":
            self.is_train = True
            file_split = cfg.data_split["train_data"]

        elif data_split == "test":
            self.is_train = False
            file_split = cfg.data_split["test_data"]

        # 读取点云数据
        self.data_dic = {}
        with open(self.pkl_path, 'rb') as f:
            self.pkl_data = pickle.load(f)
        self.modelid_data = [k for k in self.pkl_data.keys()]

        # 取两个数据集的交集
        mids = []
        for sp in file_split:
            m = np.loadtxt(sp, dtype=str).tolist()
            for i in m:
                if i in self.modelid_data:
                    mids.append(i)

        # 读取vocab映射文件
        with open('/'.join((shapenet_path, 'vocab/shapenet.json')), encoding='utf-8') as f:
            self.vocab_json = json.load(f)
        self.vocab_mapping = self.vocab_json['word_to_idx']
        self.i2w = self.vocab_json['idx_to_word']
        add_vocab(self.i2w, self.vocab_mapping, '<start>')
        add_vocab(self.i2w, self.vocab_mapping, '<end>')

        # 建立字典
        self.mid_cap_data = []  # [[mid, caption], ..]
        self.mid_cap = defaultdict(list)
        caps = self.vocab_json['captions']
        for i in caps:
            self.mid_cap[i['model']].append(i['caption'])
        for i in mids:
            for j in self.mid_cap[i]:
                self.mid_cap_data.append([i, j])

        self.length = len(self.mid_cap_data)

    def __getitem__(self, index):
        model_id, caption = self.mid_cap_data[index]
        single_sentence = caption

        # 获取点云以及语义标注
        xyz_data, _, seg_anno_data = self.pkl_data[model_id]
        choice = np.random.choice(
            seg_anno_data.shape[0], self.npoints, replace=True)
        xyz_data_ = xyz_data[choice]
        seg_anno_data_ = seg_anno_data[choice]
        if self.is_train:
            # scale
            xyz_data_[:, :3] = xyz_data_[:, :3] * np.random.uniform(0.9, 1.1)
            # rotate
            rotate_angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            rot_xyz = rotate_point(xyz_data_[:, :3], rotate_angle)
            xyz_data_[:, :3] = rot_xyz

        # normalize
        xyz_data_[:, 3:] = xyz_data_[:, 3:] - 0.5
        xyz_data_ = torch.from_numpy(xyz_data_).float()
        seg_anno_data_ = torch.from_numpy(seg_anno_data_).long()

        caption_ = []
        caption_.append(self.vocab_mapping['<start>'])
        caption_.extend([self.vocab_mapping[token] for token in caption])
        caption_.append(self.vocab_mapping['<end>'])
        target = torch.as_tensor(caption_, dtype=torch.long)

        return xyz_data_, target, seg_anno_data_, index, model_id, single_sentence

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    xyzrgbs, captions, semantic_labels, ids, model_ids, sentences = zip(*data)

    xyzrgbs = torch.stack(xyzrgbs, 0)
    semantic_labels = torch.stack(semantic_labels, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return {
        "shapes": xyzrgbs,
        "captions": targets,
        "semantic_labels": semantic_labels,
        "lengths": lengths,
        "ids": ids,
        "model_ids": model_ids,
        "sentences": sentences,
    }


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, opt.shapenet_path, vocab, opt.K, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(vocab, batch_size, workers, opt):
    dpath = opt.data_path

    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'test', vocab, opt,
                                    batch_size, True, workers, collate_fn)

    return train_loader, val_loader


## shuffle 3D shape parts
class ShufflePrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_path, data_split, shapenet_path, vocab, part_num, cfg):
        self.vocab = vocab
        # cfg for point cloud
        self.npoints = cfg.num_points
        self.pkl_path = os.path.join(shapenet_path, cfg.pkl_path)
        self.seg_num = cfg.SEG_NUM

        self.part_num = part_num

        # 读取split
        if data_split == "train":
            self.is_train = True
            file_split = cfg.data_split["train_data"]
        elif data_split == "test":
            self.is_train = False
            file_split = cfg.data_split["test_data"]

        # 读取点云数据
        self.data_dic = {}
        with open(self.pkl_path, 'rb') as f:
            self.pkl_data = pickle.load(f)
        self.modelid_data = [k for k in self.pkl_data.keys()]

        # 取两个数据集的交集
        mids = []
        for sp in file_split:
            m = np.loadtxt(sp, dtype=str).tolist()
            for i in m:
                if i in self.modelid_data:
                    mids.append(i)

        # 读取vocab映射文件
        with open('/'.join((shapenet_path, 'vocab/shapenet.json')), encoding='utf-8') as f:
            self.vocab_json = json.load(f)
        self.vocab_mapping = self.vocab_json['word_to_idx']
        self.i2w = self.vocab_json['idx_to_word']
        add_vocab(self.i2w, self.vocab_mapping, '<start>')
        add_vocab(self.i2w, self.vocab_mapping, '<end>')

        # 建立字典
        self.mid_cap_data = []  # [[mid, caption], ..]
        self.mid_cap = defaultdict(list)
        caps = self.vocab_json['captions']
        for i in caps:
            self.mid_cap[i['model']].append(i['caption'])
        for i in mids:
            for j in self.mid_cap[i]:
                self.mid_cap_data.append([i, j])

        self.length = len(self.mid_cap_data)

        # 读取part标注文件
        with open('shapenet_part_captions.json',
                  encoding='utf-8') as f:
            self.part_json = json.load(f)
        self.categories = ['03001627', '04379243']
        self.ids_2_parts = {'2': ['chair back', 'back'], '3': ['chair arm', 'arm'], '4': ['chair leg', 'leg'],
                            '5': ['chair seat', 'seat'],
                            '15': ['table top', 'top'], '16': ['table base', 'base']}
        self.caption_templates = [Template("this $shape have "), Template("it be a $shape with ")]
        self.conjunctions = [',', 'and']

    def __getitem__(self, index):
        # 获取点云以及语义标注
        xyz_data, seg_anno_data, caption = self.shuffle_parts()
        while xyz_data.shape[0] <= self.npoints:
            xyz_data, seg_anno_data, caption = self.shuffle_parts()
        single_sentence = caption

        choice = np.random.choice(
            seg_anno_data.shape[0], self.npoints, replace=True)
        xyz_data_ = xyz_data[choice]
        seg_anno_data_ = seg_anno_data[choice]
        if self.is_train:
            # scale
            xyz_data_[:, :3] = xyz_data_[:, :3] * np.random.uniform(0.9, 1.1)
            # rotate
            rotate_angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            rot_xyz = rotate_point(xyz_data_[:, :3], rotate_angle)
            xyz_data_[:, :3] = rot_xyz

        # normalize
        xyz_data_[:, 3:] = xyz_data_[:, 3:] - 0.5
        xyz_data_ = torch.from_numpy(xyz_data_).float()
        seg_anno_data_ = torch.from_numpy(seg_anno_data_).long()

        caption_ = []
        caption_.append(self.vocab_mapping['<start>'])
        caption_.extend([self.vocab_mapping[token] for token in caption])
        caption_.append(self.vocab_mapping['<end>'])
        target = torch.as_tensor(caption_, dtype=torch.long)
        return xyz_data_, target, seg_anno_data_, index, 'model_id_shuffle', single_sentence

    def __len__(self):
        return self.length

    def shuffle_parts(self):
        category = random.choice(self.categories)
        parts = []
        if category == '03001627':  # chair
            chair_back = random.choice(self.part_json[category]['chair back'])
            parts.append([chair_back, 2])

            chair_leg = random.choice(self.part_json[category]['chair leg'])
            parts.append([chair_leg, 4])

            chair_seat = random.choice(self.part_json[category]['chair seat'])
            parts.append([chair_seat, 5])

            # the chair may not have an arm
            if random.randint(0, 1) == 0:
                chair_arm = random.choice(self.part_json[category]['chair arm'])
                parts.append([chair_arm, 3])

            caption = random.choice(self.caption_templates).substitute(shape="chair").split()
        else:  # table
            table_top = random.choice(self.part_json[category]['table top'])
            parts.append([table_top, 15])

            table_base = random.choice(self.part_json[category]['table base'])
            parts.append([table_base, 16])

            caption = random.choice(self.caption_templates).substitute(shape="table").split()

        xyz_data_list = []
        seg_anno_data_list = []
        for idx in range(len(parts)):
            part_and_id = parts[idx]
            part = part_and_id[0]
            part_id = part_and_id[1]
            part_model_id = part['model_id']
            # 获取整体点云以及语义标注
            shape_xyz_data, _, shape_seg_anno_data = self.pkl_data[part_model_id]
            shape_seg_anno_data_int = shape_seg_anno_data.astype(np.int16)
            # 获取part点云以及语义标注
            part_xyz_data = shape_xyz_data[shape_seg_anno_data_int == part_id]
            part_seg_anno_data = shape_seg_anno_data[shape_seg_anno_data_int == part_id]

            xyz_data_list.append(part_xyz_data)
            seg_anno_data_list.append(part_seg_anno_data)

            # 获取part caption
            part_caption = part['part_caption']
            if self.ids_2_parts[str(part_id)][1] in part_caption:
                caption += (part_caption)
            else:
                caption += part_caption
                caption += random.choice(self.ids_2_parts[str(part_id)]).split()
            if idx != len(parts) - 1:
                caption.append(random.choice(self.conjunctions))

        adjust_margin = -0.01

        # Reposition centroid to the origin (XZ-plane as Y indicates heights)
        for index in range(len(xyz_data_list)):
            part = xyz_data_list[index]
            x_part = part[:, 0]
            z_part = part[:, 2]

            x_mean = np.mean(x_part)
            z_mean = np.mean(z_part)

            x_offset = x_part - x_mean
            z_offset = z_part - z_mean

            part[:, 0] = x_offset
            part[:, 2] = z_offset

            xyz_data_list[index] = part

        if category == '03001627':  # 是椅子
            # inter
            min_y_seat = np.min(xyz_data_list[2][:, 1])
            max_y_seat = np.max(xyz_data_list[2][:, 1])
            min_z_seat = np.min(xyz_data_list[2][:, 2])

            max_y_leg = np.max(xyz_data_list[1][:, 1])

            min_y_back = np.min(xyz_data_list[0][:, 1])
            max_z_back = np.max(xyz_data_list[0][:, 2])

            # 调整均以seat为中心
            # 调整leg
            adjust_height_leg = min_y_seat - max_y_leg - adjust_margin
            xyz_data_list[1][:, 1] += adjust_height_leg
            # 调整back
            adjust_height_back_y = min_y_back - max_y_seat - adjust_margin
            xyz_data_list[0][:, 1] -= adjust_height_back_y

            adjust_height_back_z = min_z_seat - max_z_back - adjust_margin
            xyz_data_list[0][:, 2] += adjust_height_back_z

            # intra for leg
            if_cover, distance = self.if_cover(xyz_data_list[2], xyz_data_list[1])
            if not if_cover:
                # move every point towards the origin
                intra_distance_x, intra_distance_z = self.compute_intra_distance_xz(xyz_data_list[1], distance)
                xyz_data_list[1][:, 0] += intra_distance_x
                xyz_data_list[1][:, 2] += intra_distance_z

            if len(xyz_data_list) == 4:  # 有arm
                min_y_arm = np.min(xyz_data_list[3][:, 1])
                min_z_arm = np.min(xyz_data_list[3][:, 2])
                # 调整arm
                adjust_height_arm_y = min_y_arm - max_y_seat + 0.05
                xyz_data_list[3][:, 1] -= adjust_height_arm_y

                adjust_height_arm_z = min_z_arm - min_z_seat - adjust_margin
                xyz_data_list[3][:, 2] -= adjust_height_arm_z

                # intra for arm
                if_cover, distance = self.if_cover(xyz_data_list[2], xyz_data_list[3])
                if not if_cover:
                    # move every point towards the origin
                    intra_distance_x, intra_distance_z = self.compute_intra_distance_xz(xyz_data_list[3], distance)
                    xyz_data_list[3][:, 0] += intra_distance_x
                    xyz_data_list[3][:, 2] += intra_distance_z

        else:  # 是桌子，先top后base
            # inter
            min_y_top = np.min(xyz_data_list[0][:, 1])
            max_y_base = np.max(xyz_data_list[1][:, 1])
            adjust_height = min_y_top - max_y_base - adjust_margin
            # 调整base的y轴坐标
            xyz_data_list[1][:, 1] += adjust_height

            # intra
            if_cover, distance = self.if_cover(xyz_data_list[0], xyz_data_list[1])
            if not if_cover:
                # move every point towards the origin
                intra_distance_x, intra_distance_z = self.compute_intra_distance_xz(xyz_data_list[1], distance)
                xyz_data_list[1][:, 0] += intra_distance_x
                xyz_data_list[1][:, 2] += intra_distance_z

        xyz_data = np.concatenate(xyz_data_list, axis=0)
        seg_anno_data = np.concatenate(seg_anno_data_list, axis=0)

        return xyz_data, seg_anno_data, caption

    def if_cover(self, top, base):  # on the XZ-plane as Y indicates height
        x_top = top[:, 0]
        z_top = top[:, 2]

        x_base = base[:, 0]
        z_base = base[:, 2]

        distances_top = np.sqrt(x_top ** 2 + z_top ** 2)
        distances_base = np.sqrt(x_base ** 2 + z_base ** 2)

        max_distance_top = np.max(distances_top)
        min_distance_base = np.min(distances_base)

        return max_distance_top >= min_distance_base, min_distance_base - max_distance_top

    def compute_intra_distance_xz(self, base, distance):
        x_base = base[:, 0]
        z_base = base[:, 2]
        distances = np.sqrt(x_base ** 2 + z_base ** 2)

        # in case that distances==0
        distances[distances == 0] = 1

        adjust_margin = 2.5

        unit_vectors_x = -x_base / distances * adjust_margin
        unit_vectors_z = -z_base / distances * adjust_margin

        # in case that distances==0
        unit_vectors_x[distances == 0] = 0
        unit_vectors_z[distances == 0] = 0

        # 不要越过原点
        new_distances = distances - distance * adjust_margin
        unit_vectors_x[new_distances < 0] = 0
        unit_vectors_z[new_distances < 0] = 0

        return unit_vectors_x * distance, unit_vectors_z * distance


def get_precomp_loader_shuffle(data_path, data_split, vocab, opt, batch_size=100,
                               shuffle=True, num_workers=2, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = ShufflePrecompDataset(data_path, data_split, opt.shapenet_path, vocab, opt.K, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders_shuffle(vocab, batch_size, workers, opt):
    dpath = opt.data_path

    train_loader_shuffle = get_precomp_loader_shuffle(dpath, 'train', vocab, opt,
                                                      batch_size, True, workers)

    return train_loader_shuffle
