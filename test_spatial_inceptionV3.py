from keras.applications.inception_v3 import InceptionV3
import cv2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
import keras, os, argparse
import numpy as np
from random import randint
from keras.models import load_model
import random
from scipy.stats import mode

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=1)
args = parser.parse_args()

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}

root = "D:/graduation_project/workspace/dataset/"
test = 'test'+str(args.split)
load_model_root = 'D:/graduation_project/workspace/spatial_inceptionV3_tmp/'
model_name = '9_round_spe10_epr100.h5'
detail_path = root + args.dataset + '_train_test_splits/' + test + '_spatial_detail/_frame_num.txt'
image_path = root + args.dataset + '_train_test_splits/' + test + '_spatial/'
num_classes = 51

def load_specific_data(path, name_lst):
    x = []
    y = []
    for file in name_lst:
        tmp = file.split('.')[0].split('_')
        type = int(tmp[2]) - 1
        tmpx = cv2.resize(cv2.imread(path + file), (299, 299))
        tmpy = keras.utils.to_categorical(type, num_classes)
        x.append(tmpx)
        y.append(tmpy)
    return np.array(x), np.array(y)

def get_name_lst(id, type, num):
    lst = []
    for i in range(num):
        lst.append(id + '_' + str(i) + '_' + type + '.jpg')
    return lst

def evaluate(txt_path, img_path, model):
    file = open(txt_path, 'r')
    content = file.readline()
    correct = 0
    total = 0
    while content:
        total += 1
        tmp1 = content.split(' ')
        tmp2 = tmp1[0].split('.')[0].split('_')
        num = int(tmp1[1])
        id = tmp2[0]
        type = tmp2[1]

        name_lst = get_name_lst(id, type, num)
        x, _ = load_specific_data(img_path, name_lst)

        ground_true = int(type)-1
        out = model.predict(x)
        res = np.argmax(out, axis=1)
        Mode = mode(res)[0][0]
        if Mode == ground_true:
            correct += 1

        print(tmp1[0], res, Mode, ground_true, 'correct:', correct, 'total:', total)

        content = file.readline()
    file.close()



#
# model = load_model(load_model_root + model_name)
# evaluate(detail_path, image_path, model)
# base_model = InceptionV3(weights='imagenet', include_top=False)
# evaluate(detail_path, image_path, base_model)
