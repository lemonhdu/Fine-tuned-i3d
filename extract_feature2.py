import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='rgb')
parser.add_argument('--load_model', type=str,default=r'./weights/gtea_rgb_split1_0412.pth')
parser.add_argument('--root', type=str,default=r'/root/autodl-tmp/dataset/video_frames')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_dir', type=str, default=r'/root/autodl-tmp/dataset/gtea_rgb_split1_0412')
parser.add_argument('--split',type=str,default=r'./resources/gtea_1.json')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from torch.autograd import Variable

from torchvision import transforms
import videotransforms

import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset


def run(max_steps=64e3, mode='', root='', split='./resources/gtea_1.json', batch_size=1, load_model='', save_dir=''):

    split_num_set = [0]

    for split_num in split_num_set:

        test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

        dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir,split_num=split_num)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True)

        val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir, split_num=split_num)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                     pin_memory=True)

        dataloaders = {'train': dataloader, 'val': val_dataloader}

        # setup the model
        if mode == 'flow':
            i3d = InceptionI3d(85, in_channels=2)
        else:
            i3d = InceptionI3d(85, in_channels=3)

        i3d.load_state_dict(torch.load(load_model))
        i3d.cuda()

        for phase in ['train', 'val']:
            i3d.train(False)  # Set model to evaluate mode

            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, name = data

                if len(inputs.shape) == 2:
                    continue

                print("start to ex video: "+name[0]+str(split_num))

                b, c, t, h, w = inputs.shape

                window_size = 21
                half_window = window_size // 2
                features = []
                for subscribe in range(0, t, window_size//8):
                    inputs_np = inputs.numpy()
                    inputs_shape = inputs_np.shape
                    if half_window <= subscribe <= t-half_window:
                        clip_input = inputs_np[:, :, subscribe-half_window:subscribe+half_window]

                    elif subscribe > t-half_window:
                        clip_input1 = inputs.numpy()[:, :, subscribe:t]
                        clip_input2 = np.zeros((inputs_shape[0], inputs_shape[1], half_window*2+subscribe-t, inputs_shape[3], inputs_shape[4]), dtype='float32')
                        clip_input = np.concatenate([clip_input1, clip_input2], axis=2)

                    else:
                        clip_input1 = inputs.numpy()[:, :, 0:half_window+subscribe]
                        clip_input2 = np.zeros((inputs_shape[0], inputs_shape[1], half_window-subscribe, inputs_shape[3], inputs_shape[4]), dtype='float32')
                        clip_input = np.concatenate([clip_input2, clip_input1], axis=2)

                    with torch.no_grad():
                        ip = Variable(torch.from_numpy(clip_input).cuda(), volatile=True)
                        extract_feature = i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
                        features.append(extract_feature)

                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))

                print("finish to ex video: "+name[0]+str(split_num))


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, 
        root=args.root,
        split=args.split,
        load_model=args.load_model,
        save_dir=args.save_dir)
