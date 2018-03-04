from __future__ import print_function
from sklearn.svm import SVC
from sklearn import preprocessing
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import argparse, os
from keras import regularizers
from scipy.stats import mode
import logging, time

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=1)
args = parser.parse_args()
root = "D:/graduation_project/workspace/dataset/"
train = 'train'+str(args.split)
test = 'test'+str(args.split)

layer = 'block5_pool'
train_spatial_root = root + args.dataset + '_train_test_splits/' + train + '_spatial/'+layer+'/ten1/_1x512/'
test_spatial_root = root + args.dataset + '_train_test_splits/' + test + '_spatial/'+layer+'/ten1/_1x512/'

batch_size = 256
num_classes = 51
epochs = 30
level1 = 512
level2 = 512
kjud = False
bjud = False
write_csv = False

def svc(traindata,trainlabel,testdata,testlabel):
    # beginTime = time.time()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # traindata = min_max_scaler.fit_transform(traindata)
    # testdata = min_max_scaler.transform(testdata)
    # print("transform data:", time.time() - beginTime)

    print("Start training SVM...")
    beginTime = time.time()
    svcClf = SVC(C=1,kernel="linear",cache_size=3000, max_iter=500, tol=1)
    svcClf.fit(traindata,trainlabel)
    print("SVM training time:", time.time() - beginTime)

    beginTime = time.time()
    pred_trainlabel = svcClf.predict(traindata)
    print("predict training time:", time.time() - beginTime)
    num = len(pred_trainlabel)
    print(pred_trainlabel)
    accuracy = len([1 for i in range(num) if trainlabel[i] == pred_trainlabel[i]]) / float(num)
    print("train Accuracy:", accuracy)

    beginTime = time.time()
    pred_testlabel = svcClf.predict(testdata)
    print("predict testing time:", time.time() - beginTime)
    num = len(pred_testlabel)
    print(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("test Accuracy:",accuracy)

    return svcClf


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
        tmp = tmp.replace('[', '').replace(']','')
        tmpx = [float(i) for i in tmp.split(',')]
        file_to_read.close()

        x.append(tmpx)
        y.append(int(file.split('.')[0].split('_')[1]) - 1)
        count+=1
        if count % 500 == 0:
            print(count)
    return np.array(x), np.array(y)

def preprocess_file_list(lst):
    file_list = []
    for file in lst:
        if '.txt' not in file:
            file_list.append(file)
    return file_list

rec = np.zeros((num_classes, epochs))
rec2 = np.zeros((epochs, num_classes))
acc_rec = [0] * epochs

def evaluate(model, num_epochs, detail=False):
    file_list = preprocess_file_list(os.listdir(test_spatial_root))

    correct = 0
    all = len(file_list)
    sta = [0] * 51
    cc = [0] * 51
    count = 0
    for folder in file_list:
        ground_true = int(folder.split('_')[1]) - 1
        sta[ground_true] += 1
        new_path = test_spatial_root + folder + '/'
        test_x, test_y = load_data(new_path)

        out = model.predict(test_x)
        Mode = mode(out)[0][0]
        if Mode == ground_true:
            if detail:
                print('\033[1;33;44m', count, ':', res, Mode, ground_true, '\033[0m')
            correct += 1.0
            cc[ground_true] += 1
        else:
            if detail:
                print(count, ':', res, Mode, ground_true)
        count += 1

    print(correct, all, correct / all)
    acc_rec[num_epochs] = correct / all
    Dict = {}
    for i in range(num_classes):
        rec2[num_epochs][i] = rec[i][num_epochs] = Dict[i] = cc[i] * 1.0 / sta[i]
    Dict = sorted(Dict.items(), key=lambda e: e[1], reverse=True)
    for item in Dict:
        if detail:
            print('%3d, %.3lf, %3d' % (item[0], item[1], sta[item[0]]))

    # file_to_write = open(save_name, 'w')
    # file_to_write.write(str(correct / all) + '\n')
    # for item in Dict:
    #     # file_to_write.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(sta[item[0]]) + '\n')
    #     print('%3d  %.3lf  %3d' % (item[0], item[1], sta[item[0]]), file=file_to_write)
    # file_to_write.close()



name = 'ten1_' + layer+ '_' + str(args.split) + 'keysorted_spatial_mlp_' + str(level1) + '_' + str(level2) +  '_b' + str(batch_size) + '_kr' + str(int(kjud)) + '_br'+str(int(bjud))

x_train, y_train = load_data(train_spatial_root)

x_test, y_test = load_data(test_spatial_root)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
save_ = './evaluation_statistics/' + name + '/'
if not os.path.exists(save_):
    os.makedirs(save_)

model = svc(x_train,y_train, x_test, y_test)
evaluate(model, 0)


if write_csv:
    file_to_write = open(save_+'epochs_main.csv', 'w')
    print('epochs,', end="", file=file_to_write)
    print(','.join([str(k) for k in range(epochs)]), file=file_to_write)
    print('acc,', end="", file=file_to_write)
    print(','.join([str(k) for k in acc_rec]), file=file_to_write)
    for i in range(num_classes):
        print(str(i)+',',end="", file=file_to_write)
        print(','.join([str(k) for k in rec[i]]), file=file_to_write)
    file_to_write.close()

    file_to_write = open(save_+'classes_main.csv', 'w')
    print('epochs,acc,', end="", file=file_to_write)
    print(','.join([str(k) for k in range(num_classes)]), file=file_to_write)
    for i in range(epochs):
        print(str(i) + ',', end="", file=file_to_write)
        print(str(acc_rec[i]) + ',', end="", file=file_to_write)
        print(','.join([str(k) for k in rec2[i]]), file=file_to_write)
    file_to_write.close()

    # model.save('./scripts/'+name)
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
