import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="rgb")
parser.add_argument('--root', type=str, default="/root/autodl-tmp/dataset/video_frames")
parser.add_argument('--train_split', type=str, default="./resources/gtea.json")
parser.add_argument('--save_model', type=str, default="./weights/gtea_rgb_split1_04142.pth")

args = parser.parse_args()

save_log_dir = r"/root/autodl-tmp/train_process/train_gtea_rgb_split1_04142.txt"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset import Charades as Dataset

from charades_dataset import video_index, clip_len


def run(init_lr=0.05, max_steps=1000, mode='', root='', train_split='', batch_size=6 * 1, save_model=''):
    # setup dataset
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           # videotransforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize(224)])

    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36,
                                                 pin_memory=True)

    dataloaders = {'training': dataloader, 'testing': val_dataloader}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(85)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [100, 300])  # div10 for gtea

    test_cls_added = 0
    best_cls_test = np.inf

    num_steps_per_update = 4  # accum gradient
    steps = 0
    # train it
    while steps < max_steps:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'testing']:
            if phase == 'training':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loc_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels, index, num_frame = data

                # arrange the frames to be trained
                for i, item in enumerate(index):
                    video_index[phase][item] += clip_len
                    # if the number of end frames is within clip_len, they are dropped to make batchsize uniform
                    if video_index[phase][item] > num_frame[i] - clip_len:
                        # random select the beginning frame
                        video_index[phase][item] = random.randint(1, 20)

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                # batch * cls * (n-1)/8
                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                #loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                loc_loss = F.cross_entropy(per_frame_logits.permute(0, 2, 1).contiguous().view(-1, 85), torch.max(labels, dim=1)[1].view(-1))
                predicted, label_pos = torch.max(per_frame_logits, 1)[1], torch.max(labels, dim=1)[1]
                correct = (predicted == label_pos).sum().item()
                print(correct)

                tot_loc_loss += loc_loss.item()

                loss = loc_loss

                if phase == 'training':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if num_iter == num_steps_per_update and phase == 'training':
                    steps += 1
                    # batch_size should be <= up_int(videos_train/num_steps_per_update)
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('{} Loc Loss: {:.4f} '.format(phase, tot_loc_loss / num_iter))
                    num_iter = 0

            if phase == 'testing':
                # revise at 2023.4.14
                test_cls = tot_loc_loss / num_iter
                test_cls_added += test_cls
                with open(save_log_dir, "a") as f:
                    f.write("step:" + str(steps) + " clsloss:" + str(test_cls) + "\n")

                step_interval = 15
                if steps % step_interval == 0:
                    if test_cls_added <= best_cls_test:
                        best_cls_test = test_cls_added
                        print("*****found better testing model, step is {:05d}, cls loss = {:4f} *******".format(
                                steps, best_cls_test / step_interval))
                        with open(save_log_dir, "a") as f:
                            f.write("*******save model, step is {:05d}, cls loss = {:4f} ******".format(steps, best_cls_test / step_interval) + "\n")
                        save_model_path = save_model
                        print(save_model_path)
                        torch.save(i3d.module.state_dict(), save_model_path)

                    test_cls_added = 0


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode,
        root=args.root,
        train_split=args.train_split,
        save_model=args.save_model)
