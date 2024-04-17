import cv2
import numpy as np
import os

flow_img_dir = '/disks/disk0/huang/dataset/segmentation/tlv1_flow_png'

flow_save_dir = '/disks/disk0/huang/dataset/segmentation/tlv1_flowImg_npy'

files_flow_img_dir = os.listdir(flow_img_dir)

for vid in files_flow_img_dir:

    abs_path = os.path.join(flow_img_dir,vid)

    imgs = os.listdir(abs_path)

    file_num = int(len(imgs)/2)

    cr_list = []

    read_one_img = cv2.imread(os.path.join(abs_path, imgs[0]))

    w = read_one_img.shape[1]

    h = read_one_img.shape[0]

    channels = read_one_img.shape[2]

    save_npy = np.zeros([h, w, 2, file_num], dtype=float)

    for i in range(file_num-1):
        # find flow image x
        for item in imgs:
            file_name = item.split('.')[0]
            file_name_number = int(file_name.split("_")[0])
            file_name_dim = file_name.split("_")[1]
            if file_name_number == i and file_name_dim == 'x':
                flow_img_x = item
                break

        # find flow image y
        for item in imgs:
            file_name = item.split('.')[0]
            file_name_number = int(file_name.split("_")[0])
            file_name_dim = file_name.split("_")[1]
            if file_name_number == i and file_name_dim == 'y':
                flow_img_y = item
                break

        flow_img_x_dir = os.path.join(abs_path, flow_img_x)
        flow_img_y_dir = os.path.join(abs_path, flow_img_y)

        save_npy[:, :, 0, i] = cv2.imread(flow_img_x_dir)[:, :, 0]
        save_npy[:, :, 1, i] = cv2.imread(flow_img_y_dir)[:, :, 0]

    save_dir = os.path.join(flow_save_dir, str(vid)+'.npy')

    np.save(save_dir, save_npy)




