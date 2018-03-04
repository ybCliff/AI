import argparse
import os
import numpy
import matlab.engine
import cv2
import time, logging
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
x_path = joints_root + 'partial_x/'
y_path = joints_root + 'partial_y/'
x_path_mc = joints_root + 'partial_x_mc/'
y_path_mc = joints_root + 'partial_y_mc/'
vgg_size = 224

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}

file_finally_num = Dict[args.dataset][args.scope]

if not os.path.exists(x_path):
    os.makedirs(x_path)
if not os.path.exists(y_path):
    os.makedirs(y_path)
if not os.path.exists(x_path_mc):
    os.makedirs(x_path_mc)
if not os.path.exists(y_path_mc):
    os.makedirs(y_path_mc)

file_num_list = [len(os.listdir(x_path)), len(os.listdir(y_path)), len(os.listdir(x_path_mc)), len(os.listdir(y_path_mc))]

if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(x_path_mc) and os.path.exists(y_path_mc):
    if min(file_num_list) == file_finally_num:
        print(args.dataset, args.scope,"joints_gray has been generated! exit")
        exit(0)
    else:
        args.start = max([0, min(file_num_list)-1]) if args.start == 0 else args.start
        print(args.dataset, args.scope, 'start:', args.start)



def read_matrix(path):
    file = open(path, 'r');
    content = file.read()
    file.close()
    print(path, content)
    content = [float(i) for i in content.split(',')]

    assert len(content) % args.joints_keys == 0
    matrix = numpy.array(content).reshape((int(len(content) / args.joints_keys), args.joints_keys))
    return matrix.tolist()

def write_gray_img(path, filename, img):
    assert img.ndim == 2
    if not os.path.exists(path):
        os.makedirs(path)

    file = open(path + filename, 'w')
    file.write(','.join(str(i) for i in img.reshape(vgg_size * vgg_size).tolist()))
    file.close()


if __name__ == '__main__':
    file_list = os.listdir(detail_root)
    eng = matlab.engine.start_matlab()
    print(args.scope)
    count = -1
    beginTime = time.time()
    logging.info('generate joints gray start!')
    for file in file_list:
        if file == '_frame_num.txt':
            continue
        count += 1
        if count % 100 == 0:
            print(count, time.time() - beginTime)
        if count < args.start:
            continue
        x_matrix = read_matrix(x_ + file)
        y_matrix = read_matrix(y_ + file)
        length = len(x_matrix)

        [x_matrix_mc,status1] = eng.inexact_alm_mc(x_matrix, length, args.joints_keys, nargout=2)
        [y_matrix_mc,status2] = eng.inexact_alm_mc(y_matrix, length, args.joints_keys, nargout=2)

        x_matrix = cv2.resize(numpy.array(x_matrix), (vgg_size, vgg_size))
        y_matrix = cv2.resize(numpy.array(y_matrix), (vgg_size, vgg_size))
        x_matrix_mc = cv2.resize(numpy.array(x_matrix_mc), (vgg_size, vgg_size))
        y_matrix_mc = cv2.resize(numpy.array(y_matrix_mc), (vgg_size, vgg_size))

        write_gray_img(x_path, file, x_matrix)
        write_gray_img(y_path, file, y_matrix)
        write_gray_img(x_path_mc, file, x_matrix_mc)
        write_gray_img(y_path_mc, file, y_matrix_mc)






