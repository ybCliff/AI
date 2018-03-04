'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os, argparse
import numpy as np
batch_size = 128
num_classes = 51
epochs = 12

# input image dimensions
img_rows, img_cols, img_channel = 7, 512, 1

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=2)
args = parser.parse_args()
root = "D:/graduation_project/workspace/dataset/"
train = 'train'+str(args.split)
test = 'test'+str(args.split)

train_spatial_root = root + args.dataset + '_train_test_splits/' + train + '_spatial/block5_pool/_7x7x512/'
test_spatial_root = root + args.dataset + '_train_test_splits/' + test + '_spatial/block5_pool/_7x7x512/'


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
        tmp = tmp.replace('[', '').replace(']', '').replace(' ', '')
        tmpx = [float(i) for i in tmp.split(',')]

        file_to_read.close()
        tmpx = np.reshape(np.array(tmpx), (7, 7, 512))
        x.append(sum(tmpx) / 7)
        y.append(int(file.split('.')[0].split('_')[1]) - 1)
        count+=1
        if count % 500 == 0:
            print(count)
    return np.array(x), np.array(y)

x_train, y_train = load_data(train_spatial_root)
x_test, y_test = load_data(test_spatial_root)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_channel, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_channel, img_rows, img_cols)
    input_shape = (img_channel, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channel)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channel)
    input_shape = (img_rows, img_cols, img_channel)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
