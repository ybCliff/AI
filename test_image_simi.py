# import cv2
# import numpy as np
# import sys
#
#
# def rgb2gray(rgb):
#     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     return gray
#
# def pHash(img):
#     """get image pHash value"""
#     #加载并调整图片为32x32灰度图片
#     img = rgb2gray(img)
#     img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
#
#         #创建二维列表
#     h, w = img.shape[:2]
#     vis0 = np.zeros((h,w), np.float32)
#     vis0[:h,:w] = img       #填充数据
#
#     #二维Dct变换
#     vis1 = cv2.dct(cv2.dct(vis0))
#     #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
#     vis1.resize(32,32)
#
#     #把二维list变成一维list
#     img_list = vis1.flatten().tolist()
#
#     #计算均值
#     avg = sum(img_list)*1./len(img_list)
#     avg_list = ['0' if i<avg else '1' for i in img_list]
#
#     #得到哈希值
#     return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,32*32,4)])
#
# def hammingDist(s1, s2):
#     assert len(s1) == len(s2)
#     return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
#
# def cal_simi(img1, img2):
#     s1 = pHash(img1)
#     s2 = pHash(img2)
#     return 1 - hammingDist(s1, s2) * 1. / (32 * 32 / 4)
#
# if __name__ == '__main__':
#     path1 = "D:/graduation_project/workspace/dataset/HMDB51_train_test_splits/train1_spatial/1100_8_16.jpg"
#     path2 = "D:/graduation_project/workspace/dataset/HMDB51_train_test_splits/train1_spatial/1100_9_16.jpg"
#     img1 = cv2.imread(path1)
#     img2 = cv2.imread(path2)
#     print(cal_simi(img1, img2))
# -*- coding: utf-8 -*-
# feimengjuan
# 利用python实现多种方法来实现图像识别

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 最简单的以灰度直方图作为相似比较的实现
def classify_gray_hist(image1, image2, size=(256, 256)):
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
def classify_hist_with_split(image1, image2, size=(256, 256)):
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


# 平均哈希算法计算
def classify_aHash(image1, image2):
    image1 = cv2.resize(image1, (8, 8))
    image2 = cv2.resize(image2, (8, 8))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1, hash2)


def classify_pHash(image1, image2):
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


if __name__ == '__main__':
    path1 = "D:/graduation_project/workspace/dataset/HMDB51_train_test_splits/train1_failed/1950_8_28.jpg"
    path2 = "D:/graduation_project/workspace/dataset/HMDB51_train_test_splits/train1_spatial/1950_9_28.jpg"
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    degree = classify_gray_hist(img1, img2)
    print(degree)
    degree = classify_hist_with_split(img1,img2)
    print(degree)
    degree = classify_aHash(img1,img2)
    print(degree)
    degree = classify_pHash(img1,img2)
    print(degree)
    cv2.waitKey(0)