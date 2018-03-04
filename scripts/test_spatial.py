import cv2
import time
import os, sys, shutil
import random
target_folder = 'D:/graduation_project/workspace/dataset/UCF101_train_test_splits/train3_spatial/'
file_list = os.listdir(target_folder)
file_list = sorted(file_list)
print(file_list[0:100])

time_start = time.time()

N = 25  # 25 frames
# v = 9537 # train1
# v = 9586 # train2
v = 9624 # train3
video_jud = [0] * v
frame_jud = [1] * N

omit = []
pre = -1
for i in file_list:
    tmp1 = i.split('_')
    video_id = int(tmp1[0])
    frame_id = int(tmp1[1])
    if video_id != pre:
        if sum(frame_jud) != N:
            print(pre, frame_jud)
            omit.append(pre)
        frame_jud = [0] * N
    pre = video_id
    video_jud[video_id] = 1
    frame_jud[frame_id] = 1

video_omit = []
if sum(video_jud) != v:
    print('OH NO!')
    for i in range(v):
        if video_jud[i] == 0:
            video_omit.append(i)

print(omit)
print(video_omit)
print(len(omit))