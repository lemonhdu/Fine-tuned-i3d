import cv2
import os
import numpy as np
import glob
import multiprocessing

######calculate flow from video frames
video_root = './video_list.txt'
root = '/disks/disk0/huangxvfeng/dataset/segmentation/50salads_full'
out_root = '/disks/disk0/huangxvfeng/dataset/segmentation/50salads_flow_jpg'


def cal_for_frames(video_path, flow_path):

    if not os.path.exists(flow_path):
        os.mkdir(os.path.join(flow_path))

    frames = glob.glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    # flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    print(video_path)
    for i, frame_curr in enumerate(frames[1:]):
        print(i)
        if os.path.exists(os.path.join(flow_path, str(i+1).zfill(5) + '_x.jpg')):
            continue
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        cv2.imwrite(os.path.join(flow_path, str(i+1).zfill(5) + '_x.jpg'), tmp_flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, str(i+1).zfill(5) + '_y.jpg'), tmp_flow[:, :, 1])
        prev = curr

    # return flow


def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    if not os.path.exists(flow_path):
        os.mkdir(os.path.join(flow_path))
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, str(i).zfill(5) + '_x.jpg'), flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, str(i).zfill(5) + '_y.jpg'), flow[:, :, 1])


def process(video_path, flow_path):
    cal_for_frames(video_path, flow_path)
    # flow = cal_for_frames(video_path, flow_path)
    # save_flow(flow, flow_path)


def extract_flow(root, out_root):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    # dir_list = os.listdir(root)
    dir_list = []
    ### read video information from .txt files
    with open(video_root, 'r') as f:
        for id, line in enumerate(f):
            video_name = line.strip().split()
            preffix = video_name[0].split('.')[0]
            dir_list.append(preffix)

    pool = multiprocessing.Pool(processes=4)
    for dir_name in dir_list:
        video_path = os.path.join(root, dir_name)
        flow_path = os.path.join(out_root, dir_name)

        # flow = cal_for_frames(video_path)
        # save_flow(flow,flow_path)
        # print('save flow data: ',flow_path)
        # process(video_path,flow_path)
        pool.apply_async(process, args=(video_path, flow_path))

    pool.close()
    pool.join()


if __name__ == '__main__':
    extract_flow(root, out_root)
    print("finish!!!!!!!!!!!!!!!!!!")

