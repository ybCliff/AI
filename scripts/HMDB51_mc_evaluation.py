import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse
import os
from enum import Enum
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow.python.client import timeline

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=1)
args = parser.parse_args()

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}

root = "D:/graduation_project/workspace/dataset/HMDB51/"
video_path = root + "video/"

x_path = root + 'ori_x/'
y_path = root + 'ori_y/'
x_path_mc = root + 'ori_x_mc/'
y_path_mc = root + 'ori_y_mc/'


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NMS_Threshold = 0.1
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1
Min_Subset_Cnt = 4
Min_Subset_Score = 0.8
Max_Human = 96

def is_zero(x_lst, y_lst, index):
    if x_lst[index] == 0 and y_lst[index] == 0:
        return True
    else:
        return False


def draw_img(img, x_lst, y_lst):
    img_copied = np.copy(img)
    centers = {}
    for i in range(CocoPart.Background.value):
        if x_lst[i] == 0 and y_lst[i] == 0:
            continue
        center = (int(x_lst[i] + 0.5), int(y_lst[i] + 0.5))
        centers[i] = center
        cv2.circle(img_copied, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

    # draw line
    for pair_order, pair in enumerate(CocoPairsRender):
        if is_zero(x_lst, y_lst, pair[0]) or is_zero(x_lst, y_lst, pair[1]):
            continue

        img_copied = cv2.line(img_copied, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return img_copied

joints_keys = 18


def load_data(fn):
    file = open(fn, 'r')
    content = file.read()
    content = content.split('\n')
    tlen = len(content)
    if tlen == 1:
        content = [float(i) for i in content[0].split(',')]
        return np.reshape(np.array(content), (len(content) // joints_keys, joints_keys))
    else:
        while(content[len(content)-1] == ""):
            content.pop()
        for i in range(len(content)):
            content[i] = [float(k) for k in content[i].split(',')]
        return np.array(content)

def display(img, img_mc):
    image_h, image_w = img.shape[:2]
    scale = 480.0 / image_w
    newh1, neww1 = int(scale * image_h + 0.5), 480
    img = cv2.resize(img, (neww1, newh1), interpolation=cv2.INTER_AREA)

    image_h, image_w = img_mc.shape[:2]
    scale = 480.0 / image_w
    newh2, neww2 = int(scale * image_h + 0.5), 480
    img_mc = cv2.resize(img_mc, (neww2, newh2), interpolation=cv2.INTER_AREA)

    convas = np.zeros([newh1 + newh2, 480, 3], dtype=np.uint8)
    convas[:newh1, :] = img
    convas[newh1:, :] = img_mc

    cv2.imshow('result', convas)
    cv2.waitKey(0)

video_name = "2740_23.avi"
# file_list = os.listdir(train_x_path)
fn = video_name.split('.')[0] + '.txt'
x = load_data(x_path+fn)
y = load_data(y_path+fn)
x_mc = load_data(x_path_mc+fn)
y_mc = load_data(y_path_mc+fn)

vc = cv2.VideoCapture(video_path + video_name)  # 读入视频文件
if not vc.isOpened():
    print('Open failure! exit')
    exit(0)
total = vc.get(cv2.CAP_PROP_FRAME_COUNT)

count = -1
rval = True
while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    count += 1
    if frame is not None:
        img = draw_img(frame, x[count], y[count])
        img_mc = draw_img(frame, x_mc[count], y_mc[count])
        print(img.shape, img_mc.shape)
        display(img, img_mc)
    cv2.waitKey(1)
vc.release()
