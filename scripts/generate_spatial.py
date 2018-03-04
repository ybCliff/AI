import cv2
import time
import os, sys, shutil
import random
source_folder = 'D:/graduation_project/workspace/dataset/UCF101_train_test_splits/train3/'
target_folder = 'D:/graduation_project/workspace/dataset/UCF101_train_test_splits/train3_spatial/'
file_list = os.listdir(source_folder)
time_start = time.time()
N = 33  # 25 frames
omit = []

def splist(l, n):
    length = len(l)
    sz = length // n
    c = length % n
    lst = []
    i = 0
    while i < n:
        if i < c:
            bs = sz + 1
            lst.append(l[i*bs:i*bs+bs])
        else:
            lst.append(l[i*sz+c:i*sz+c+sz])
        i += 1
    return lst

def run(video, N):
    tmp1 = video.split('.')
    tmp2 = tmp1[0].split('_')
    status = True
    vc = cv2.VideoCapture(source_folder + video)  # 读入视频文件

    if not vc.isOpened():
        print('Open failure! exit')
        exit(0)

    total = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    sp = splist([i for i in range(int(total))], N)
    print('hahaha ',video, tmp2[0], total)

    lst = []
    for i in range(N):
        lst.append(random.sample(sp[i], 1))

    index = 0
    count = 0
    rval = True
    while rval and index < N:  # 循环读取视频帧
        rval, frame = vc.read()
        if (count == lst[index][0]):
            cv2.imwrite(target_folder
                        + tmp2[0] + '_'
                        + str(index) + '_'
                        + tmp2[1] + '.jpg',
                        frame)  # 存储为图像
            if frame is None:
                print(video, index, lst[index][0])
                print(lst)
                status = False
            index += 1
        count += 1
        cv2.waitKey(1)
    vc.release()
    # if index < N:
    #     return False
    return status


debugC = 0
for video in file_list:
    tmp1 = video.split('.')
    tmp2 = tmp1[0].split('_')

    if len(omit) > 0:
        if int(tmp2[0]) in omit:
            status = run(video, N)
            while status is False:
                status = run(video, N)
    else:
        if debugC >= 0:
            status = run(video, N)
            while status is False:
                status = run(video, N)

    if debugC % 100 == 0:
        print(debugC)
    debugC += 1

print(time.time()-time_start, 'ms')
















