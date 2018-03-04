import cv2
import time
import os, sys, shutil
import random
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Generate spatial frame from video')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--scope', type=str, default='test1')
parser.add_argument('--detail', type=bool, default=True)
parser.add_argument('--failed', type=bool, default=True)
args = parser.parse_args()

root = 'D:/graduation_project/workspace/dataset/' + args.dataset + '_train_test_splits/'
source_folder = root + args.scope + '/'
target_folder = root + args.scope + '_spatial/'
detail_path = root + args.scope + '_spatial_detail/'
failed_path = root + args.scope + '_failed/'

file_list = os.listdir(source_folder)
time_start = time.time()
N = 10  # 25 frames
omit = []
similarity_lower_bound = 0.43
dist_upper_bound = 15

if not os.path.exists(target_folder):
    os.makedirs(target_folder)
if not os.path.exists(detail_path):
    os.makedirs(detail_path)
if args.failed and not os.path.exists(failed_path):
    os.makedirs(failed_path)

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}
file_finally_num = Dict[args.dataset][args.scope]
if len(os.listdir(detail_path)) - 1 == file_finally_num:
    print(args.dataset, args.scope,"spatial has been generated! exit")
    exit(0)



def cal_simi_pHash(image1, image2):
    image1 = cv2.resize(image1, (32, 32))
    image2 = cv2.resize(image2, (32, 32))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1, hash2)

def cal_simi_aHash(image1,image2):
    image1 = cv2.resize(image1,(8,8))
    image2 = cv2.resize(image2,(8,8))
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1,hash2)

# 输入灰度图，返回hash
def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

def cal_simi_hist_gray(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

# 计算单通道的直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


# 通过得到每个通道的直方图来计算相似度
def cal_simi_hist_rgb(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

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

def check_image_similarity(frame_list):
    length = len(frame_list)
    fail_count = [0] * length
    fail_count_limit = int(0.7 * length)
    fail_list = []
    for i in range(length):
        for j in range(i+1, length):
            if max(cal_simi_hist_rgb(frame_list[i], frame_list[j]), cal_simi_hist_gray(frame_list[i], frame_list[j])) < similarity_lower_bound \
                    and min(cal_simi_pHash(frame_list[i], frame_list[j]), cal_simi_aHash(frame_list[i], frame_list[j])) > dist_upper_bound:
                fail_count[i] += 1
                fail_count[j] += 1
    for i in range(length):
        if fail_count[i] >= fail_count_limit:
            fail_list.append(i)
    return fail_list

def run(video, N):
    tmp1 = video.split('.')
    tmp2 = tmp1[0].split('_')
    vc = cv2.VideoCapture(source_folder + video)  # 读入视频文件

    if not vc.isOpened():
        print('Open failure! exit')
        exit(0)

    total = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    sp = splist([i for i in range(int(total-1))], N)

    lst = []
    for i in range(N):
        lst.append(random.sample(sp[i], 1))

    index = 0
    count = 0
    rval = True
    rec_frame = []
    rec_path = []
    while rval and index < N:  # 循环读取视频帧
        rval, frame = vc.read()
        if count == lst[index][0] and frame is not None:
            if frame is None:
                lst[index][0] += 1
            else:
                rec_frame.append(frame)
                rec_path.append(failed_path
                            + tmp2[0] + '_'
                            + str(index) + '_'
                            + tmp2[1] + '.jpg')
                index += 1
        count += 1
        cv2.waitKey(1)
    vc.release()

    assert len(rec_frame) == N
    failed = check_image_similarity(rec_frame)
    # print(video, failed)
    if args.detail:
        file_to_write = open(detail_path + tmp1[0] + '.txt', 'w')
        tmp = []
        for i in range(N):
            if i not in failed:
               tmp.append(str(lst[i][0]))
        file_to_write.write(' '.join(tmp))
        file_to_write.close()

    index = 0
    for i in range(N):
        if i not in failed:
            cv2.imwrite(target_folder
                        + tmp2[0] + '_'
                        + str(index) + '_'
                        + tmp2[1] + '.jpg', rec_frame[i])
            index += 1

    if len(failed) > 0 and args.failed:
        for key in failed:
            cv2.imwrite(rec_path[key], rec_frame[key])

    return N - len(failed)


debugC = 0
if args.detail:
    file = open(detail_path + '_frame_num.txt', 'w')
for video in file_list:
    tmp1 = video.split('.')
    tmp2 = tmp1[0].split('_')

    if len(omit) > 0:
        if int(tmp2[0]) in omit:
            tmp = run(video, N)
    else:
        tmp = run(video, N)
        if args.detail:
            file.write(video + ' ' + str(tmp) + '\n')
    if debugC % 100 == 0:
        print(debugC)
    debugC += 1

if args.detail:
    file.close()
print(time.time()-time_start, 'ms')
















