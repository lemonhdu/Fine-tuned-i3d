import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path
import torch.nn.functional as F
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
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(5)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    # modified in 2.25
    img = cv2.resize(img,dsize=(224,224))
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(5)+'_x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(5)+'_y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    if imgx is None:
        imgx = cv2.imread(os.path.join(image_dir, vid, str(i-1).zfill(5)+'_x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, str(i-1).zfill(5)+'_y.jpg'), cv2.IMREAD_GRAYSCALE)
        
    w,h = imgx.shape

    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
    
    imgx, imgy = cv2.resize(imgx,dsize=(224,224)), cv2.resize(imgy,dsize=(224,224))
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=85):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames//2
            
        if num_frames < 66:
            continue

        label = np.zeros((num_classes,num_frames), np.float32)

        fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0, num_frames, 1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0], fr] = 1 # binary classification

        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1
        
    return dataset

video_index = {"training": [1 for i in range(21)], "testing": [1 for i in range(7)]}
clip_len = 64

class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None):
        super(Charades, self).__init__()
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.split = split

    def __getitem__(self, index):

        vid, label, dur, nf = self.data[index]

        start_f = video_index[self.split][index]  # start_f = random.randint(1, nf-65)

        global clip_len

        num_frame = clip_len

        if start_f + num_frame >= nf:
            print("Finish video"+str(index))

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, num_frame)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, num_frame)

        label = label[:, start_f:start_f+num_frame]
        
        # imgs = np.repeat(imgs, 8, axis=0)
        #
        # label = np.repeat(label, 8, axis=1)
        
        # imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), index, nf

    def __len__(self):
        return len(self.data)
