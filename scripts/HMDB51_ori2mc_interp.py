import argparse
import os
import numpy
import matlab.engine
import cv2
import time
from matplotlib import pyplot as plt
from scipy import interpolate

root = "D:/graduation_project/workspace/dataset/HMDB51/"
x_path = root + 'ori_x/'
y_path = root + 'ori_y/'
x_path_mc = root + 'ori_x_mc_nearest/'
y_path_mc = root + 'ori_y_mc_nearest/'

if not os.path.exists(x_path_mc):
    os.makedirs(x_path_mc)
if not os.path.exists(y_path_mc):
    os.makedirs(y_path_mc)



def write_gray_img(path, filename, img):
    if not os.path.exists(path):
        os.makedirs(path)

    w,h=img.shape
    for i in range(w):
        for j in range(h):
            if img[i, j] < 1e-7:
                img[i, j] = 0

    file = open(path + filename, 'w')
    for i in range(w):
        print(','.join([str(i) for i in img[i]]), file=file)
    file.close()


def load_data(fn):
    file_to_read = open(fn, 'r')
    content = file_to_read.read()
    content = content.split('\n')
    while content[len(content) - 1] == '':
        content.pop()
    for i in range(len(content)):
        content[i] = [float(k) for k in content[i].split(',')]
    return content

def get_xy(matrix, key):
    w = matrix.shape[0]
    x = []
    y = []
    interest = []
    for i in range(w):
        if matrix[i,key] == 0:
            interest.append(i)
        else:
            x.append(i)
            y.append(matrix[i, key])
    return x, y, interest, len(interest)/w

def interp(x, y, xx, kind="nearest"):#'nearest', 'zero', 'linear', 'quadratic'
    if(len(x) < 2):
        return [], []
    # print(x)
    # print(xx)
    while len(xx) > 0 and (xx[len(xx)-1] > x[len(x)-1]):
        xx.pop()
    while len(xx) > 0 and (xx[0] < x[0]):
        xx.pop(0)
    f = interpolate.interp1d(x, y, kind=kind)
    yy = f(xx)  # 计算插值结果
    return xx, yy

def get_mc_matrix(matrix, joints_keys=18):
    for i in range(joints_keys):
        x, y, xx, _ = get_xy(matrix, i)
        xx, yy = interp(x, y, xx)
        for j in range(len(xx)):
            matrix[xx[j]][i] = yy[j]
    return matrix

if __name__ == '__main__':
    file_list = os.listdir(x_path)
    count = 0
    beginTime = time.time()
    # start = max([0, len(os.listdir(x_path_mc)) - 1])
    start = 0
    for file in file_list:
        count += 1
        if count % 100 == 0:
            print(count, time.time() - beginTime)
        if count < start:
            continue
        if file == '_frame_num.txt':
            continue

        x_matrix = load_data(x_path+file)
        y_matrix = load_data(y_path+file)

        x_matrix_mc = get_mc_matrix(numpy.array(x_matrix))
        y_matrix_mc = get_mc_matrix(numpy.array(y_matrix))

        write_gray_img(x_path_mc, file, x_matrix_mc)
        write_gray_img(y_path_mc, file, y_matrix_mc)







