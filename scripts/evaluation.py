import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras import regularizers
import numpy as np
import os, argparse
from scipy.stats import mode
import logging

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--mc', type=bool, default=True)
args = parser.parse_args()

root = "D:/graduation_project/workspace/dataset/"
test = 'test'+str(args.split)

test_spatial_root = root + args.dataset + '_train_test_splits/' + test + '_spatial/block5_pool/_1x512/'

def preprocess_file_list(lst):
    file_list = []
    for file in lst:
        if '.txt' not in file:
            file_list.append(file)
    return file_list

def load_data(path):
    x = []
    y = []
    file_list = os.listdir(path)
    count = 0
    for file in file_list:
        if '.txt' not in file:
            continue
        file_to_read = open(path + file, 'r')
        tmp = file_to_read.read()
        tmpx = [float(i) for i in tmp.split(',')]
        file_to_read.close()

        x.append(tmpx)
        y.append(int(file.split('.')[0].split('_')[1]) - 1)
        count+=1
        if count % 500 == 0:
            print(count)
    return np.array(x), np.array(y)

model_name = 'spatial_mlp_1024_512_e30_b256.h5'
model = load_model(model_name)

file_list = preprocess_file_list(os.listdir(test_spatial_root))

correct = 0
all = len(file_list)
sta = [0] * 51
cc = [0] * 51
count = 0
for folder in file_list:
    ground_true = int(folder.split('_')[1])-1
    sta[ground_true] += 1
    new_path = test_spatial_root + folder + '/'
    test_x, test_y = load_data(new_path)

    out = model.predict(test_x)
    res = np.argmax(out, axis=1)
    Mode = mode(res)[0][0]
    if Mode == ground_true:
        print('\033[1;33;44m', count, ':', res, Mode, ground_true, '\033[0m')
        correct += 1.0
        cc[ground_true] += 1
    else:
        print(count, ':', res, Mode, ground_true)
    count += 1

print(correct, all, correct / all)
Dict = {}
for i in range(51):
    Dict[i] = cc[i] * 1.0 / sta[i]
Dict = sorted(Dict.items(), key=lambda e:e[1], reverse=True)
for item in Dict:
    print('%3d, %.3lf, %3d' %(item[0], item[1], sta[item[0]]))

pre = model_name.split('.')[0]
file_to_write = open('../evaluation_statistics/'+pre+'.txt', 'w')
file_to_write.write(pre + '\n')
file_to_write.write(str(correct / all) + '\n')
for item in Dict:
    # file_to_write.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(sta[item[0]]) + '\n')
    print('%3d  %.3lf  %3d' % (item[0], item[1], sta[item[0]]), file=file_to_write)
file_to_write.close()