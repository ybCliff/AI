'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras import regularizers
import numpy as np
import os, argparse

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--mc', type=bool, default=True)
args = parser.parse_args()

root = "D:/graduation_project/workspace/dataset/"
train = 'train'+str(args.split)
test = 'test'+str(args.split)

train_joints_root = root + args.dataset + '_train_test_splits/' + train + '_joints/block5_pool/'
test_joints_root = root + args.dataset + '_train_test_splits/' + test + '_joints/block5_pool/'

train_x_path = train_joints_root + 'partial_x_mc/' if args.mc else train_joints_root + 'partial_x/'
train_y_path = train_joints_root + 'partial_y_mc/' if args.mc else train_joints_root + 'partial_y/'

test_x_path = test_joints_root + 'partial_x_mc/' if args.mc else test_joints_root + 'partial_x/'
test_y_path = test_joints_root + 'partial_y_mc/' if args.mc else test_joints_root + 'partial_y/'


def load_data(x_path, y_path):
    x = []
    y = []
    file_list = os.listdir(x_path)
    for file in file_list:
        file_to_read = open(x_path + file, 'r')
        tmp = file_to_read.read()
        x1 = [float(i) for i in tmp.split(',')]
        file_to_read.close()

        file_to_read = open(y_path + file, 'r')
        tmp = file_to_read.read()
        x2 = [float(i) for i in tmp.split(',')]
        file_to_read.close()

        x.append(x1+x2)
        y.append(int(file.split('.')[0].split('_')[1]) - 1)

    return np.array(x), np.array(y)

batch_size = 128
num_classes = 51
epochs = 2000

x_train, y_train = load_data(train_x_path, train_y_path)
x_test, y_test = load_data(test_x_path, test_y_path)

print(x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
a = max(x_train.tolist())
print(a)
a = np.mat(a)
print(a[a>0])
print((a[a>0]).shape)
# convert class vectors to binary class matrices
print(y_train[0])
y_train = keras.utils.to_categorical(y_train, num_classes)
print(y_train[0])
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1024,)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

# model.summary()
mode = load_model('dummy.h5')
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))
model.save('dummy2.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
