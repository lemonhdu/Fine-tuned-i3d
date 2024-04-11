import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, end_num):
    frames = []
    for i in range(start, end_num + 1):
        # img_path = os.path.join(image_dir, vid, str(i).zfill(5)+'.jpg')
        # img = cv2.imread(img_path)
        # if img is None:
        #     continue
        # else:
        #     img = img[:, :, [2, 1, 0]]
        i = i + 1
        img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(5) + '.jpg'))
        if img is None:
            img = cv2.imread(os.path.join(image_dir, vid, str(i - 1).zfill(5) + '.jpg'))
        img = img[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = cv2.resize(img, dsize=(224, 224))
        img = (img / 255.) * 2 - 1
        frames.append(img)
        if i % 1000 == 0:
            print(i)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, end_num):
    frames = []
    for i in range(start, end_num + 1):

        imgx = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(5) + '_x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(5) + '_y.jpg'), cv2.IMREAD_GRAYSCALE)

        if imgx is None:
            imgx = cv2.imread(os.path.join(image_dir, vid, str(i - 1).zfill(5) + '_x.jpg'), cv2.IMREAD_GRAYSCALE)
            imgy = cv2.imread(os.path.join(image_dir, vid, str(i - 1).zfill(5) + '_y.jpg'), cv2.IMREAD_GRAYSCALE)
        w, h = imgx.shape

        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx, imgy = cv2.resize(imgx, dsize=(224, 224)), cv2.resize(imgy, dsize=(224, 224))
        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


# split_path, "training", frames_path, "rgb"
def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)  # 从json文件中读取数据

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames // 2

        label = np.zeros((num_classes, num_frames), np.float32)

        fps = num_frames / data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0, num_frames, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[ann[0], fr] = 1  # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1

    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0, split_num=0):

        self.data = make_dataset(split_file, split, root, mode)  # split_path, "training", frames_path, "rgb"
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        self.split_num = split_num

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]

        # 一次加载5000图片
        low_nf = 9000 * self.split_num + 1
        up_nf = 9000 * (self.split_num + 1)
        if up_nf >= nf:
            up_nf = nf

        # 如果下标越界
        if low_nf > nf:
            return torch.tensor([0.0]), torch.tensor([0.0]), vid

        if up_nf - low_nf <= 64:
            low_nf = up_nf - 64
            print("No enough images for last split" + vid + "_" + str(self.split_num))

        # 如果已经存在
        if os.path.exists(os.path.join(self.save_dir, vid + "_" + str(self.split_num) + '.npy')):
            return torch.tensor([0.0]), torch.tensor([0.0]), vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, low_nf, up_nf)
        else:
            imgs = load_flow_frames(self.root, vid, low_nf, up_nf)

        # imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
