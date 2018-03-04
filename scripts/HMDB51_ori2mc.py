import argparse
import os
import numpy
import matlab.engine
import cv2
import time
from matplotlib import pyplot as plt

root = "D:/graduation_project/workspace/dataset/HMDB51/"
x_path = root + 'ori_x/'
y_path = root + 'ori_y/'
x_path_mc = root + 'ori_x_mc/'
y_path_mc = root + 'ori_y_mc/'

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
    file.write(','.join(str(i) for i in img.reshape(w*h).tolist()))
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

if __name__ == '__main__':
    file_list = os.listdir(x_path)
    eng = matlab.engine.start_matlab()
    count = 0
    beginTime = time.time()
    start = max([0, len(os.listdir(x_path_mc)) - 1])
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

        print(numpy.array(x_matrix).shape)
        [x_matrix_mc,status1] = eng.inexact_alm_mc(x_matrix, len(x_matrix), 18, nargout=2)
        [y_matrix_mc,status2] = eng.inexact_alm_mc(y_matrix, len(y_matrix), 18, nargout=2)

        x_matrix_mc = numpy.array(x_matrix_mc)
        y_matrix_mc = numpy.array(y_matrix_mc)

        write_gray_img(x_path_mc, file, x_matrix_mc)
        write_gray_img(y_path_mc, file, y_matrix_mc)







