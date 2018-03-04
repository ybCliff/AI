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

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=1)
args = parser.parse_args()

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}

root = "D:/graduation_project/workspace/dataset/"
train = 'train'+str(args.split)
test = 'test'+str(args.split)
file_finally_num = Dict[args.dataset][test]

train_spatial_root = root + args.dataset + '_train_test_splits/' + train + '_spatial/'
test_spatial_root = root + args.dataset + '_train_test_splits/' + test + '_spatial/'

batch_size = 64
num_classes = 51
epochs = 800

def preprocess_file_list(lst):
    res = []
    for file in lst:
        if '.jpg' in file:
            res.append(file)
    return res

def load_evaluation_data(path):
    print('loading evaluation data')
    file_list = preprocess_file_list(os.listdir(path))
    random.shuffle(file_list)
    vis = [0] * file_finally_num
    x = []
    y = []
    for file in file_list:
        tmp = file.split('.')[0].split('_')
        id = int(tmp[0])
        type = int(tmp[2]) - 1
        if vis[id] == 0:
            vis[id] = 1
            tmpx = cv2.resize(cv2.imread(path + file), (299, 299))
            tmpy = keras.utils.to_categorical(type, num_classes)
            x.append(tmpx)
            y.append(tmpy)
    print('loading done!')
    return np.array(x), np.array(y)

x_test, y_test = load_evaluation_data(test_spatial_root)

def generate_batch_testdata_random(batch_size):
    ylen = len(y_test)
    count = ylen // batch_size
    if count * batch_size == ylen:
        count -= 1
    while (True):
        i = randint(0,count)
        yield x_test[i * batch_size:min([(i + 1) * batch_size, ylen])], y_test[i * batch_size:min([(i + 1) * batch_size, ylen])]


def generate_batch_traindata_random(path, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ori_file_list = preprocess_file_list(os.listdir(path))
    random.shuffle(ori_file_list)
    print(ori_file_list)
    ylen = len(ori_file_list)
    count = ylen // batch_size
    if count * batch_size == ylen:
        count -= 1
    while (True):
        i = randint(0, count)
        file_list = ori_file_list[i * batch_size:min([(i + 1) * batch_size, ylen])]
        x = []
        y = []
        for file in file_list:
            tmpx = cv2.resize(cv2.imread(path + file), (299, 299))
            tmpy = int(file.split('.')[0].split('_')[2]) - 1
            tmpy = keras.utils.to_categorical(tmpy, num_classes)
            # print(tmpx.shape)
            x.append(tmpx)
            y.append(tmpy)
        x = np.array(x)
        y = np.array(y)
        # print(x.shape)
        yield x, y


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# model = load_model('1_round.h5')

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
save_model_root = 'D:/graduation_project/workspace/spatial_inceptionV3_tmp/'
spe = 10
epochs_per_round = 100
round = 10
for i in range(round):
    model.fit_generator(generate_batch_traindata_random(train_spatial_root, batch_size),
        samples_per_epoch=spe, epochs=epochs_per_round,
        validation_data=generate_batch_testdata_random(batch_size),
        validation_steps=1,
        verbose=1)
    model.save(str(i)+'_round_spe'+str(spe)+'_epr'+str(epochs_per_round)+'.h5')

