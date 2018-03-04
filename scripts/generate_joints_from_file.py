import argparse
import os
import numpy
import matlab.engine
import cv2
import time
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Generate joints from file')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--scope', type=str, default='train1')
parser.add_argument('--joints_keys', type=int, default=18)
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

root = "D:/graduation_project/workspace/dataset/"
joints_root = root + args.dataset + '_train_test_splits/' + args.scope + '_joints/'
detail_root = root + args.dataset + '_train_test_splits/' + args.scope + '_spatial_detail/'
x_ = joints_root + 'x/'
y_ = joints_root + 'y/'
# x_path = joints_root + 'partial_x/'
# y_path = joints_root + 'partial_y/'
x_path_mc = joints_root + 'ori_x_mc/'
y_path_mc = joints_root + 'ori_y_mc/'
vgg_size = 224
if not os.path.exists(x_path_mc):
    os.makedirs(x_path_mc)
if not os.path.exists(y_path_mc):
    os.makedirs(y_path_mc)

def read_specific_frame(path, lst):
    file = open(path, 'r');
    matrix = []
    content = file.read()
    file.close()
    content = content.split('\n')
    clen = len(content)
    for k in range(clen):
        tmp = [float(i) for i in content[k].split(',')]
        matrix.append(tmp)
    return matrix, clen

def write_gray_img(path, filename, img):
    if not os.path.exists(path):
        os.makedirs(path)

    w,h=img.shape
    file = open(path + filename, 'w')
    file.write(','.join(str(i) for i in img.reshape(w*h).tolist()))
    file.close()

def sparse(matrix):
    w, h = matrix.shape
    not_zero_count = 0
    for i in range(w):
        for j in range(h):
            if matrix[i, j] != 0:
                not_zero_count += 1
    return not_zero_count * 1.0 / (w * h)

if __name__ == '__main__':
    file_list = os.listdir(detail_root)
    eng = matlab.engine.start_matlab()
    #
    # for file in file_list:
    #     if file is '_frame_num.txt':
    #         continue
    #     file_to_read = open(file, 'r')
    #     content = file_to_read.read()
    #     print(content)
    print(args.scope)
    count = -1
    beginTime = time.time()
    x_sparse = [0.0] * 51
    x_count = [0] * 51
    y_sparse = [0.0] * 51
    y_count = [0] * 51
    tx = [0.0] * 51
    ty = [0.0] * 51
    for file in file_list:
        count += 1
        if count % 100 == 0:
            print(count, time.time() - beginTime)
        if count < args.start:
            continue
        # file = "1019_15.txt"
        file_to_read = open(detail_root + file, 'r')
        content = file_to_read.read()
        lst = [int(i) for i in content.split(' ')]
        x_matrix, clen = read_specific_frame(x_ + file, lst)
        y_matrix, _ = read_specific_frame(y_ + file, lst)
        print(numpy.array(x_matrix))

        [x_matrix_mc,status1] = eng.inexact_alm_mc(x_matrix, clen, args.joints_keys, nargout=2)
        [y_matrix_mc,status2] = eng.inexact_alm_mc(y_matrix, clen, args.joints_keys, nargout=2)

        # x_matrix = cv2.resize(numpy.array(x_matrix), (vgg_size, vgg_size))
        # y_matrix = cv2.resize(numpy.array(y_matrix), (vgg_size, vgg_size))
        # x_matrix_mc = cv2.resize(numpy.array(x_matrix_mc), (vgg_size, vgg_size))
        # y_matrix_mc = cv2.resize(numpy.array(y_matrix_mc), (vgg_size, vgg_size))
        x_matrix_mc = numpy.array(x_matrix_mc)
        y_matrix_mc = numpy.array(y_matrix_mc)

        # write_gray_img(x_path, file, x_matrix)
        # write_gray_img(y_path, file, y_matrix)
        write_gray_img(x_path_mc, file, x_matrix_mc)
        write_gray_img(y_path_mc, file, y_matrix_mc)

        type = int(file.split('.')[0].split('_')[1]) - 1
        xsp = sparse(numpy.array(x_matrix))
        ysp = sparse(numpy.array(y_matrix))
        x_sparse[type] += xsp
        y_sparse[type] += ysp
        x_count[type] += 1
        y_count[type] += 1

        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(tmpX)
        # plt.subplot(2, 2, 2)
        # plt.imshow(tmpY)
        # plt.subplot(2, 2, 3)
        # plt.imshow(tmpXmc)
        # plt.subplot(2, 2, 4)
        # plt.imshow(tmpYmc)
        # plt.show()

    for i in range(51):
        tx[i] = x_sparse[i] / x_count[i]
        ty[i] = y_sparse[i] / y_count[i]

    save = '../evaluation_statistics/sparse_joints_mc/'
    if not os.path.exists(save):
        os.makedirs(save)

    file_to_write = open(save + args.scope + '.csv', 'w')
    print(','.join([str(i) for i in range(51)]), file=file_to_write)
    res = [0.0] * 51
    for i in range(51):
        res[i] = 0.5 * (tx[i] + ty[i])
    print(','.join([str(i) for i in res]), file=file_to_write)
    file_to_write.close()
    # file = open(x_path_mc+'_failed.txt', 'w')
    # file.write('\n'.join([str(i) for i in x_failed]))
    # file.close()
    # file = open(y_path_mc+'_failed.txt', 'w')
    # file.write('\n'.join([str(i) for i in y_failed]))
    # file.close()





