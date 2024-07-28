import os
import torch
import numpy as np
import json

import cv2
from PIL import Image

from torch.utils.data import Dataset

import sys
import argparse
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import h5py
import os
import pycocotools._mask as _mask

import torch
from torch.autograd import Variable


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]


def prepare_relations(n):
    node_r_idx = np.arange(n)
    node_s_idx = np.arange(n)

    rel = np.zeros((n**2, 2))
    rel[:, 0] = np.repeat(np.arange(n), n)
    rel[:, 1] = np.tile(np.arange(n), n)

    # print(rel)

    n_rel = rel.shape[0]
    Rr_idx = torch.LongTensor([rel[:, 0], np.arange(n_rel)])
    Rs_idx = torch.LongTensor([rel[:, 1], np.arange(n_rel)])
    value = torch.FloatTensor([1] * n_rel)

    rel = [Rr_idx, Rs_idx, value, node_r_idx, node_s_idx]

    return rel


def convert_mask_to_bbox(mask, H, W, bbox_size):
    h, w = mask.shape[0], mask.shape[1]
    x = np.repeat(np.arange(h), w).reshape(h, w)
    y = np.tile(np.arange(w), h).reshape(h, w)
    x = np.sum(mask * x) / np.sum(mask) * (float(H) / h)
    y = np.sum(mask * y) / np.sum(mask) * (float(W) / w)
    bbox = int(x - bbox_size / 2), int(y - bbox_size / 2), bbox_size, bbox_size
    ret = np.ones((2, bbox_size, bbox_size))
    ret[0, :, :] *= x
    ret[1, :, :] *= y
    return bbox, torch.FloatTensor(ret)


def crop(src, bbox, H, W):
    x, y, h, w = bbox
    # print(bbox)
    shape = list(src.shape)
    shape[0], shape[1] = h, w
    ret = np.zeros(shape)

    x_ = max(-x, 0)
    y_ = max(-y, 0)
    x = max(x, 0)
    y = max(y, 0)
    h_ = min(h - x_, H - x)
    w_ = min(w - y_, W - y)

    # print(x, y, x_, y_, h_, w_)

    ret[x_:x_+h_, y_:y_+w_] = src[x:x+h_, y:y+w_]

    # print(src[x:x+h, y:y+w])
    # cv2.imshow('img', np.array(ret * 255, dtype=np.uint8))
    # cv2.waitKey(0)
    return torch.FloatTensor(ret)


def encode_attr(material, shape, bbox_size, attr_dim):
    attr = np.zeros(attr_dim)
    if material == 'rubber':
        attr[0] = 1
    elif material == 'metal':
        attr[1] = 1
    else:
        raise AssertionError("unknown material: " + material)

    if shape == 'cube':
        attr[2] = 1
    elif shape == 'cylinder':
        attr[3] = 1
    elif shape == 'sphere':
        attr[4] = 1
    else:
        raise AssertionError("unknown shape: " + shape)

    ret = np.ones((bbox_size, bbox_size, attr_dim)) * attr
    ret = np.swapaxes(ret, 0, 2)

    return torch.FloatTensor(ret)


def normalize(x, mean, std):
    return (x - mean) / std


def check_attr(id):
    color, material, shape = id
    if material == 'metal' or material == 'rubber':
        pass
    else:
        raise AssertionError("unknown material: " + material)

    if shape == 'cube' or shape == 'sphere' or shape == 'cylinder':
        pass
    else:
        raise AssertionError("unknown shape: " + shape)


def get_identifier(obj):
    color = obj['color']
    material = obj['material']
    shape = obj['shape']
    return color, material, shape


def get_identifiers(objects):
    ids = []
    for i in range(len(objects)):
        id = get_identifier(objects[i])
        check_attr(id)
        ids.append(id)
    return ids


def check_same_identifier(id_0, id_1):
    len_id = len(id_0)
    for i in range(len_id):
        if id_0[i] != id_1[i]:
            return False
    return True


def check_contain_id(id, ids):
    for i in range(len(ids)):
        if check_same_identifier(id, ids[i]):
            return True
    return False


def check_same_identifiers(ids_0, ids_1):
    len_ids = len(ids_0)
    for i in range(len_ids):
        find_same_id = False
        for j in range(len_ids):
            if check_same_identifier(ids_0[i], ids_1[j]):
                find_same_id = True
                break
        if not find_same_id:
            return False

    return True


def get_masks(objects):
    masks = []
    for i in range(len(objects)):
        mask = decode(objects[i]['mask'])
        masks.append(mask)
    return masks


def check_valid_masks(masks):
    for i in range(len(masks)):
        if np.sum(masks[i]) == 0:
            return False
    return True


def check_duplicate_identifier(objects):
    n_objects = len(objects)
    for xx in range(n_objects):
        id_xx = get_identifier(objects[xx])
        for yy in range(xx + 1, n_objects):
            id_yy = get_identifier(objects[yy])
            if check_same_identifier(id_xx, id_yy):
                return True
    return False


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def norm(x):
    return np.sqrt(np.sum(x**2))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_variable(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor),
                        requires_grad=requires_grad)


def sort_by_x(obj):
    return obj[1][0, 1, 0, 0]


def merge_img_patch(img_0, img_1):
    # cv2.imshow('img_0', img_0.astype(np.uint8))
    # cv2.imshow('img_1', img_1.astype(np.uint8))

    ret = img_0.copy()
    idx = img_1[:, :, 0] > 0
    idx = np.logical_or(idx, img_1[:, :, 1] > 0)
    idx = np.logical_or(idx, img_1[:, :, 2] > 0)
    ret[idx] = img_1[idx]

    # cv2.imshow('ret', ret.astype(np.uint8))
    # cv2.waitKey(0)

    return ret


def make_video(filename, frames, H, W, bbox_size, back_ground=None, store_img=False):

    n_frame = len(frames)

    # print('states', states.shape)
    # print('actions', actions.shape)
    # print(filename)

    # print(actions[:, 0, :])
    # print(states[:20, 0, :])

    videoname = filename + '.avi'
    os.system('mkdir -p ' + filename)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}

    colors = [np.array([255,160,122]),
              np.array([224,255,255]),
              np.array([216,191,216]),
              np.array([255,255,224]),
              np.array([245,245,245]),
              np.array([144,238,144])]

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, 3, (W, H))

    if back_ground is not None:
        bg = cv2.imread(back_ground)
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    for i in range(n_frame):
        objs, rels, feats = frames[i]
        n_objs = len(objs)

        if back_ground is not None:
            frame = bg.copy()
        else:
            frame = np.ones((H, W, 3), dtype=np.uint8) * 255

        objs = objs.copy()

        # obj: attr, [mask_crop, pos, img_crop], id
        objs.sort(key=sort_by_x)

        n_object = len(objs)
        for j in range(n_object):
            obj = objs[j][1][0]

            mask = obj[:1].permute(1, 2, 0).data.numpy()
            img = obj[3:].permute(1, 2, 0).data.numpy()
            mask = np.clip((mask + 0.5) * 255, 0, 255)
            img = np.clip((img * 0.5 + 0.5) * mask, 0, 255)
            # img *= mask

            n_rels = len(rels)
            collide = False
            for k in range(n_rels):
                id_0, id_1 = rels[k][0], rels[k][1]
                if check_same_identifier(id_0, objs[j][2]) or check_same_identifier(id_1, objs[j][2]):
                    collide = True

            if collide:
                _, cont, _ = cv2.findContours(
                    mask.astype(np.uint8)[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cont, -1, (0, 255, 0), 1)

                '''
                print(i, j)
                cv2.imshow('mask', mask.astype(np.uint8))
                cv2.imshow('img', img.astype(np.uint8))
                cv2.waitKey(0)
                '''

            if np.isnan(obj[1, 0, 0]) or np.isnan(obj[2, 0, 0]):
                # check if the position is NaN
                continue
            if np.isinf(obj[1, 0, 0]) or np.isinf(obj[2, 0, 0]):
                # check if the position is inf
                continue

            x = int(obj[1, 0, 0] * H/2. + H/2. - bbox_size/2)
            y = int(obj[2, 0, 0] * W/2. + W/2. - bbox_size/2)

            # print(x, y, H, W)
            h, w = int(bbox_size), int(bbox_size)
            x_ = max(-x, 0)
            y_ = max(-y, 0)
            x = max(x, 0)
            y = max(y, 0)
            h_ = min(h - x_, H - x)
            w_ = min(w - y_, W - y)

            # print(x, y, x_, y_, h_, w_)

            if x + h_ < 0 or x >= H or y + w_ < 0 or y >= W:
                continue

            frame[x:x+h_, y:y+w_] = merge_img_patch(
                frame[x:x+h_, y:y+w_], img[x_:x_+h_, y_:y_+w_])

        if store_img:
            cv2.imwrite(os.path.join(filename, 'img_%d.png' % i), frame.astype(np.uint8))
        # cv2.imshow('img', frame.astype(np.uint8))
        # cv2.waitKey(0)

        out.write(frame)

    out.release()


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [N, D]
        # y: [M, D]
        x = x.repeat(y.size(0), 1, 1)   # x: [M, N, D]
        x = x.transpose(0, 1)           # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)   # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)    # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)
def collate_fn(data):
    return data[0]


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class PhysicsCLEVRDataset(Dataset):

    def __init__(self, args, phase,data_dir):
        self.args = args
        self.phase = phase
        self.loader = default_loader
        self.data_dir = data_dir
        self.label_dir = args.label_dir
        self.valid_idx_lst = 'valid_idx_' + self.phase + '.txt'
        self.H = 100
        self.W = 150
        self.bbox_size = 24

        ratio = self.args.train_valid_ratio
        n_train = round(self.args.n_rollout * ratio)
        if phase == 'train':
            self.st_idx = 0
            self.n_rollout = n_train
        elif phase == 'valid':
            self.st_idx = n_train
            self.n_rollout = self.args.n_rollout - n_train
        else:
            raise AssertionError("Unknown phase")

        if self.args.gen_valid_idx:
            self.gen_valid_idx()
        else:
            self.read_valid_idx()

    def read_valid_idx(self):
        # if self.phase == 'train':
        # return
        print("Reading valid idx ...")
        self.n_valid_idx = 0
        self.valid_idx = []
        self.metadata = []
        fin = open(self.valid_idx_lst, 'r').readlines()

        self.n_valid_idx = len(fin)
        for i in range(self.n_valid_idx):
            a = int(fin[i].strip().split(' ')[0])
            b = int(fin[i].strip().split(' ')[1])
            self.valid_idx.append((a, b))

        for i in range(self.st_idx, self.st_idx + self.n_rollout):
            if i % 500 == 0:
                print("Reading valid idx %d/%d" % (i, self.st_idx + self.n_rollout))

            with open(os.path.join(self.label_dir, 'sim_%05d.json' % i)) as f:
                data = json.load(f)
                self.metadata.append(data)

    def gen_valid_idx(self):
        print("Preprocessing valid idx ...")
        self.n_valid_idx = 0
        self.valid_idx = []
        self.metadata = []
        fout = open(self.valid_idx_lst, 'w')

        n_his = self.args.n_his
        frame_offset = self.args.frame_offset

        for i in range(self.st_idx, self.st_idx + self.n_rollout):
            if i % 500 == 0:
                print("Preprocessing valid idx %d/%d" % (i, self.st_idx + self.n_rollout))
            
            with open(os.path.join(self.label_dir, 'sim_%05d.json' % i)) as f:
                data = json.load(f)
                self.metadata.append(data)

            gt = data['ground_truth']
            gt_ids = gt['objects']
            gt_collisions = gt['collisions']

            for j in range(
                n_his * frame_offset,
                len(data['frames']) - frame_offset):

                objects = data['frames'][j]['objects']
                n_object_cur = len(objects)
                identifiers_cur = get_identifiers(objects)
                valid = True

                # check whether the current frame is valid:
                if check_duplicate_identifier(objects):
                    valid = False

                '''
                masks = get_masks(objects)
                if not check_valid_masks(masks):
                    valid = False
                '''

                # check whether history window is valid
                for k in range(n_his):
                    idx = j - (k + 1) * frame_offset
                    objects = data['frames'][idx]['objects']
                    n_object = len(objects)
                    identifiers = get_identifiers(objects)
                    # masks = get_masks(objects)

                    if (not valid) or n_object != n_object_cur:
                        valid = False
                        break
                    if not check_same_identifiers(identifiers, identifiers_cur):
                        valid = False
                        break
                    if check_duplicate_identifier(objects):
                        valid = False
                        break

                    '''
                    if not check_valid_masks(masks):
                        valid = False
                        break
                    '''

                # check whether the target is valid
                idx = j + frame_offset
                objects_nxt = data['frames'][idx]['objects']
                n_object_nxt = len(objects_nxt)
                identifiers_nxt = get_identifiers(objects_nxt)
                if n_object_nxt != n_object_cur:
                    valid = False
                elif not check_same_identifiers(identifiers_nxt, identifiers_cur):
                    valid = False
                elif check_duplicate_identifier(objects_nxt):
                    valid = False

                # check if detected the right objects for collision
                for k in range(len(gt_collisions)):
                    if 0 <= gt_collisions[k]['frame'] - j < frame_offset:
                        gt_obj = gt_collisions[k]['object']

                        id_0 = gt_obj[0]
                        id_1 = gt_obj[1]
                        for t in range(len(gt_ids)):
                            if id_0 == gt_ids[t]['id']:
                                id_x = get_identifier(gt_ids[t])
                            if id_1 == gt_ids[t]['id']:
                                id_y = get_identifier(gt_ids[t])

                        # id_0 = get_identifier(gt_ids[gt_obj[0]])
                        # id_1 = get_identifier(gt_ids[gt_obj[1]])
                        if not check_contain_id(id_x, identifiers_cur):
                            valid = False
                        if not check_contain_id(id_y, identifiers_cur):
                            valid = False

                '''
                masks_nxt = get_masks(objects_nxt)
                if not check_valid_masks(masks_nxt):
                    valid = False
                '''

                if valid:
                    self.valid_idx.append((i - self.st_idx, j))
                    fout.write('%d %d\n' % (i - self.st_idx, j))
                    self.n_valid_idx += 1

        fout.close()

    '''
    def read_valid_idx(self):
        fin = open(self.valid_idx_lst, 'r').readlines()
        self.n_valid_idx = len(fin)
        self.valid_idx = []
        for i in range(len(fin)):
            idx = [int(x) for x in fin[i].strip().split(' ')]
            self.valid_idx.append((idx[0], idx[1]))
    '''

    def __len__(self):
        return self.n_valid_idx

    def __getitem__(self, idx):
        n_his = self.args.n_his
        frame_offset = self.args.frame_offset
        idx_video, idx_frame = self.valid_idx[idx][0], self.valid_idx[idx][1]

        objs = []
        attrs = []
        for i in range(
            idx_frame - n_his * frame_offset,
            idx_frame + frame_offset + 1, frame_offset):

            frame = self.metadata[idx_video]['frames'][i]
            frame_filename = frame['frame_filename']
            objects = frame['objects']
            n_objects = len(objects)

            img = self.loader(os.path.join(self.data_dir, frame_filename))
            img = np.array(img)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.

            ### prepare object inputs
            object_inputs = []
            for j in range(n_objects):
                material = objects[j]['material']
                shape = objects[j]['shape']

                if i == idx_frame - n_his * frame_offset:
                    attrs.append(encode_attr(
                        material, shape, self.bbox_size, self.args.attr_dim))

                mask_raw = decode(objects[j]['mask'])
                mask = cv2.resize(mask_raw, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                # cv2.imshow("mask", mask * 255)
                # cv2.waitKey(0)

                bbox, pos = convert_mask_to_bbox(mask_raw, self.H, self.W, self.bbox_size)
                # print(pos)

                pos_mean = torch.FloatTensor(np.array([self.H / 2., self.W / 2.]))
                pos_mean = pos_mean.unsqueeze(1).unsqueeze(1)
                pos_std = pos_mean

                pos = normalize(pos, pos_mean, pos_std)
                # print(pos)
                mask_crop = normalize(crop(mask, bbox, self.H, self.W), 0.5, 1).unsqueeze(0)
                img_crop = normalize(crop(img, bbox, self.H, self.W), 0.5, 0.5).permute(2, 0, 1)

                identifier = get_identifier(objects[j])

                # print(torch.max(pos), torch.min(pos))
                # print('mask_crop size', mask_crop.size())
                # print('pos size', pos.size())
                # print('img_crop size', img_crop.size())

                s = torch.cat([mask_crop, pos, img_crop], 0).unsqueeze(0), identifier
                object_inputs.append(s)

            objs.append(object_inputs)

        attr = torch.cat(attrs, 0).view(
            n_objects, self.args.attr_dim, self.bbox_size, self.bbox_size)

        feats = []
        for x in range(n_objects):
            feats.append(objs[0][x][0])

        for i in range(1, len(objs)):
            for x in range(n_objects):
                for y in range(n_objects):
                    id_x = objs[0][x][1]
                    id_y = objs[i][y][1]
                    if check_same_identifier(id_x, id_y):
                        feats[x] = torch.cat([feats[x], objs[i][y][0]], 1)

        # for i in range(1, self.args.state_dim * (n_his + 2), self.args.state_dim):
        # print(feats[0][0, i, 0, 0], feats[0][0, i, 1, 1])
        # print()

        try:
            feats = torch.cat(feats, 0)
        except:
            print(idx_video, idx_frame)
        # print("feats shape", feats.size())

        ### prepare relation attributes
        n_relations = n_objects * n_objects
        Ra = torch.FloatTensor(
            np.ones((
                n_relations,
                self.args.relation_dim * (self.args.n_his + 2),
                self.bbox_size,
                self.bbox_size)) * -0.5)

        # change to relative position
        relation_dim = self.args.relation_dim
        state_dim = self.args.state_dim
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # x
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # y

        # add collision attr
        gt = self.metadata[idx_video]['ground_truth']
        gt_ids = gt['objects']
        gt_collisions = gt['collisions']

        label_rel = torch.FloatTensor(np.ones((n_objects * n_objects, 1)) * -0.5)

        if self.args.edge_superv:
            for i in range(
                idx_frame - n_his * frame_offset,
                idx_frame + frame_offset + 1, frame_offset):

                for j in range(len(gt_collisions)):
                    frame_id = gt_collisions[j]['frame']
                    if 0 <= frame_id - i < self.args.frame_offset:
                        id_0 = gt_collisions[j]['object'][0]
                        id_1 = gt_collisions[j]['object'][1]
                        for k in range(len(gt_ids)):
                            if id_0 == gt_ids[k]['id']:
                                id_x = get_identifier(gt_ids[k])
                            if id_1 == gt_ids[k]['id']:
                                id_y = get_identifier(gt_ids[k])

                        # id_0 = get_identifier(gt_ids[gt_collisions[j]['object'][0]])
                        # id_1 = get_identifier(gt_ids[gt_collisions[j]['object'][1]])

                        for k in range(n_objects):
                            if check_same_identifier(objs[0][k][1], id_x):
                                x = k
                            if check_same_identifier(objs[0][k][1], id_y):
                                y = k

                        idx_rel_xy = x * n_objects + y
                        idx_rel_yx = y * n_objects + x

                        # print(x, y, n_objects)

                        idx = i - (idx_frame - n_his * frame_offset)
                        idx /= frame_offset
                        Ra[idx_rel_xy, int(idx) * relation_dim] = 0.5
                        Ra[idx_rel_yx, int(idx) * relation_dim] = 0.5

                        if i == idx_frame + frame_offset:
                            label_rel[idx_rel_xy] = 1
                            label_rel[idx_rel_yx] = 1

        '''
        print(feats[0, -state_dim])
        print(feats[0, -state_dim+1])
        print(feats[0, -state_dim+2])
        print(feats[0, -state_dim+3])
        print(feats[0, -state_dim+4])
        '''

        '''
        ### change absolute pos to relative pos
        feats[:, state_dim+1::state_dim] = \
                feats[:, state_dim+1::state_dim] - feats[:, 1:-state_dim:state_dim]   # x
        feats[:, state_dim+2::state_dim] = \
                feats[:, state_dim+2::state_dim] - feats[:, 2:-state_dim:state_dim]   # y
        feats[:, 1] = 0
        feats[:, 2] = 0
        '''

        x = feats[:, :-state_dim]
        label_obj = feats[:, -state_dim:]
        label_obj[:, 1] -= feats[:, -2*state_dim+1]
        label_obj[:, 2] -= feats[:, -2*state_dim+2]
        rel = prepare_relations(n_objects)
        rel.append(Ra[:, :-relation_dim])

        '''
        print(rel[-1][0, 0])
        print(rel[-1][0, 1])
        print(rel[-1][0, 2])
        print(rel[-1][2, 3])
        print(rel[-1][2, 4])
        print(rel[-1][2, 5])
        '''

        # print("attr shape", attr.size())
        # print("x shape", x.size())
        # print("label_obj shape", label_obj.size())
        # print("label_rel shape", label_rel.size())

        '''
        for i in range(n_objects):
            print(objs[0][i][1])
            print(label_obj[i, 1])

        time.sleep(10)
        '''

        return attr, x, rel, label_obj, label_rel