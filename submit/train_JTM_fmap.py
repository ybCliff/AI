import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, LSTM, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
import argparse
import numpy as np
import os, time, cv2
from scipy.stats import mode

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--model', type=str, default='vgg19')
parser.add_argument('--layer', type=str, default='block5_pool')
parser.add_argument('--fusion', type=str, default='mean')   # mean  or  max
parser.add_argument('--folder', type=str, default='JTM_mc/20')   # JTM_mc/x, JTM_ori/x, JDM_mc/x, JDM_ori/x, spatial_x/frame
parser.add_argument('--split', type=int, default=1)
args = parser.parse_args()

spatial_num = 20
loop_num = 5
assert spatial_num % loop_num == 0

num_classes = 51
batch_size = 512
epochs = 100
read_loop = True
alternate = False

root = "D:/graduation_project/workspace/dataset/"+args.dataset+'/'
folder_name = args.model + '_' + args.layer + '_' + args.fusion
train = "train" + str(args.split)
test = "test" + str(args.split)
fmap_train_path = root + train + '/' + args.folder + '/' + folder_name + '/'
fmap_test_path = root + test + '/' + args.folder + '/' + folder_name + '/'

def load_data(path):

    if not os.path.exists(path):
        exit(0)
    file_list = os.listdir(path)
    print(path, len(file_list))
    x = []
    y = []
    count = 0
    beginTime = time.time()
    myset = []
    for file in file_list:
        count += 1
        if count % 500 == 0:
            print(count, time.time()-beginTime)
        file_to_read = open(path+file, 'r')
        content = file_to_read.read()
        file_to_read.close()

        content = [float(i) for i in content.split(',')]
        tmpy = file.split('.')[0].split('_')[2]

        if read_loop:
            myset.append(content)
            if count % loop_num == 0:
                x.append(myset)
                y.append(int(tmpy) - 1)
                myset = []
        else:
            x.append(content)
            y.append(int(tmpy) - 1)

    return np.array(x), np.array(y)

def alternate_load_data(path1, path2):
    if not os.path.exists(path1) or not os.path.exists(path2):
        exit(100)
    file_list = os.listdir(path1)
    print(path1, path2, len(file_list))
    x = []
    y = []
    count = 0
    beginTime = time.time()
    myset1 = []
    myset2 = []

    for file in file_list:
        count += 1
        if count % 500 == 0:
            print(count, time.time()-beginTime)
        file_to_read = open(path1+file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content = [float(i) for i in content.split(',')]
        myset1.append(content)

        file_to_read = open(path2 + file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content = [float(i) for i in content.split(',')]
        myset2.append(content)

        if count % loop_num == 0:
            tmpy = file.split('.')[0].split('_')[2]
            x.append(np.append(myset1, myset2, axis=0))
            y.append(int(tmpy) - 1)
            myset1 = []
            myset2 = []

    return np.array(x), np.array(y)


def mean_max_load(path1, path2):
    if not os.path.exists(path1) or not os.path.exists(path2):
        exit(100)
    file_list = os.listdir(path1)
    print(path1, path2, len(file_list))
    x = []
    y = []
    count = 0
    beginTime = time.time()
    myset1 = []
    for file in file_list:
        count += 1
        if count % 500 == 0:
            print(count, time.time()-beginTime)
        file_to_read = open(path1+file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content1 = [float(i) for i in content.split(',')]

        file_to_read = open(path2 + file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content2 = [float(i) for i in content.split(',')]
        tmp = np.append(content1, content2, axis=0)

        myset1.append(np.mean(tmp, axis=0))

        if count % loop_num == 0:
            tmpy = file.split('.')[0].split('_')[2]
            x.append(myset1)
            y.append(int(tmpy) - 1)
            myset1 = []

    return np.array(x), np.array(y)

def mlp(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(512,)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(x_test, y_test))


def cnn_lstm(x_train, y_train, x_test, y_test):
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 128

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    # print(x_train.ndim)
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(512,1)))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


def lstm(x_train, y_train, x_test, y_test):
    drop = 0.4
    model = Sequential()
    model.add(LSTM(128, input_shape=(loop_num * 2 if alternate else loop_num, 512), return_sequences=True, dropout=drop))
    # model.add(LSTM(512, return_sequences=True, dropout=drop))
    model.add(LSTM(64, dropout=drop))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=2)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return model


def evaluate1(model, x, y, intervel=5):
    begin = 0
    correct = 0
    all = 0
    print("Begin to evaluate")
    while begin < x.shape[0]:
        ground_true = np.where(y[begin]==1)[0][0]
        tmpx = x[begin:begin+intervel]
        tmpy = model.predict(tmpx)
        res = np.argmax(tmpy, axis=1)
        Mode = mode(res)[0][0]
        begin += intervel
        all += 1
        if Mode == ground_true:
            correct += 1.0
    print("Accuracy:", correct/all)


def evaluate2(model, x, y, intervel=spatial_num//loop_num, level1=0, level2=0, pre=0):
    begin = 0
    correct = 0
    all = 0
    type_count = [0] * 51
    correct_count = [0] * 51
    # print("Begin to evaluate")
    while begin < x.shape[0]:
        ground_true = np.where(y[begin]==1)[0][0]
        tmpx = x[begin:begin+intervel]
        tmpy = model.predict(tmpx)

        tmp = np.zeros((1, num_classes))
        tmp[tmp==0] = 1.0
        for k in range(intervel):
            tmp *= tmpy[k]

        res = np.argmax(tmp, axis=1)
        begin += intervel
        all += 1

        type_count[ground_true] += 1
        if res[0] == ground_true:
            correct_count[ground_true] += 1
            correct += 1.0
    print("Accuracy:", correct/all)
    res = []
    res_count = []
    for i in range(51):
        if type_count[i] == 0:
            continue
        res.append(correct_count[i] / type_count[i])
        res_count.append(type_count[i])
        # print(i, type_count[i], correct_count[i] / type_count[i])

    if correct/all > pre:
        file_to_write = open(str(spatial_num)+ '_b' + str(batch_size) + '_'+str(level1) + '_' + str(level2) + '.csv', 'w')
        file_to_write.write(','.join([str(i) for i in res])+'\n')
        file_to_write.write(','.join([str(i) for i in res_count]) + '\n')
        file_to_write.write(str(correct/all))
        file_to_write.close()
    return correct/all

def evaluate3(model, x, y, intervel=5):
    begin = 0
    correct = 0
    all = 0
    print("Begin to evaluate")
    while begin < x.shape[0]:
        ground_true = np.where(y[begin]==1)[0][0]
        tmpx = x[begin:begin+intervel]
        tmpy = model.predict(tmpx)

        res = np.max(tmpy, axis=0)
        res = np.argmax(res, axis=0)
        begin += intervel
        all += 1
        if res == ground_true:
            correct += 1.0
    print("Accuracy:", correct/all)


def loop_test(x_train, y_train, x_test, y_test):
    drop = 0.4
    rec = {}
    trg_path = '../model/'
    if not os.path.exists(trg_path):
        os.makedirs(trg_path)
    # for level1 in range(320, 1025, 64):
    #     for level2 in range(64, level1+1, 64):
    for level1 in range(128, 129, 64):
        for level2 in range(64, 65, 64):
            print(level1,'_',level2)
            model = Sequential()
            model.add(LSTM(level1, input_shape=(loop_num * 2 if alternate else loop_num, 512), return_sequences=True, dropout=drop))
            # model.add(LSTM(512, return_sequences=True, dropout=drop))
            model.add(LSTM(level2, dropout=drop))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer=RMSprop(),
                          metrics=['accuracy'])

            pre_best = 0
            best_model = Sequential()
            best_epochs = 0
            for i in range(epochs):
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=1,
                          validation_data=(x_test, y_test),
                          verbose=2)
                tmp = evaluate2(model, x_test, y_test, level1=level1, level2=level2, pre=pre_best)
                if tmp > pre_best:
                    pre_best = tmp
                    best_model = model
                    best_epochs = i
            # best_model.save(trg_path + 'b' + str(batch_size) + '_' + str(level1) + '_' + str(level2) + '_' + str(int(pre_best * 10000))+'_e' + str(best_epochs)+ '.h5')
            rec[str(level1)+'_'+str(level2)]=pre_best
    print(rec)
    for item in rec:
        print(item)

if __name__ == '__main__':
    # train_mc_path = root + train + '/JTM_mc/25/' + folder_name + '/'
    # train_ori_path = root + train + '/JTM_ori/25/' + folder_name + '/'
    # test_mc_path = root + test + '/JTM_mc/25/' + folder_name + '/'
    # test_ori_path = root + test + '/JTM_ori/25/' + folder_name + '/'
    #
    # x_train, y_train = alternate_load_data(train_mc_path, train_ori_path)
    # x_test, y_test = alternate_load_data(test_mc_path, test_ori_path)
    x_train, y_train = load_data(fmap_train_path)
    x_test, y_test = load_data(fmap_test_path)
    print("train samples shape:", x_train.shape)
    print("test samples shape:", x_test.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    loop_test(x_train, y_train, x_test, y_test)
    # model = lstm(x_train, y_train, x_test, y_test)
    # evaluate1(model, x_test, y_test)
    # evaluate2(model, x_test, y_test)
    # evaluate3(model, x_test, y_test)
    # mlp(x_train, y_train, x_test, y_test)
    # cnn_lstm(x_train, y_train, x_test, y_test)
