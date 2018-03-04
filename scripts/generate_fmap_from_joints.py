from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import argparse
import numpy as np
import os, time
parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--scope', type=str, default='train2')
parser.add_argument('--layer', type=str, default='block5_pool')
parser.add_argument('--top', type=bool, default=False)
parser.add_argument('--gray', type=bool, default=False)
args = parser.parse_args()
root = "D:/graduation_project/workspace/dataset/"
joints_root = root + args.dataset + '_train_test_splits/' + args.scope + '_joints/'
x_path = joints_root + 'partial_x/'
y_path = joints_root + 'partial_y/'
x_path_mc = joints_root + 'partial_x_mc/'
y_path_mc = joints_root + 'partial_y_mc/'

x_path_w = joints_root + args.layer + '/partial_x/'
y_path_w = joints_root + args.layer + '/partial_y/'
x_path_mc_w = joints_root + args.layer + '/partial_x_mc/'
y_path_mc_w = joints_root + args.layer + '/partial_y_mc/'


Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}

file_finally_num = Dict[args.dataset][args.scope]

if not os.path.exists(x_path_w):
    os.makedirs(x_path_w)
if not os.path.exists(y_path_w):
    os.makedirs(y_path_w)
if not os.path.exists(x_path_mc_w):
    os.makedirs(x_path_mc_w)
if not os.path.exists(y_path_mc_w):
    os.makedirs(y_path_mc_w)

status1 = status2 = status3 = status4 = True
start1 = max([0, len(os.listdir(x_path_w))-1])
start2 = max([0, len(os.listdir(y_path_w))-1])
start3 = max([0, len(os.listdir(x_path_mc_w))-1])
start4 = max([0, len(os.listdir(y_path_mc_w))-1])

if len(os.listdir(x_path_w)) == file_finally_num:
    print(args.dataset, args.scope,"partial_x joints_fmap has been generated!")
    status1 = False
if len(os.listdir(y_path_w)) == file_finally_num:
    print(args.dataset, args.scope,"partial_y joints_fmap has been generated!")
    status2 = False
if len(os.listdir(x_path_mc_w)) == file_finally_num:
    print(args.dataset, args.scope,"partial_x_mc joints_fmap has been generated!")
    status3 = False
if len(os.listdir(y_path_mc_w)) == file_finally_num:
    print(args.dataset, args.scope,"partial_y_mc joints_fmap has been generated!")
    status4 = False




vgg_size = 224

jetMap = np.array([[0.0, 0.0, 0.51563], [0.0, 0.0, 0.53125], [0.0, 0.0, 0.54688], [0.0, 0.0, 0.5625], [0.0, 0.0, 0.57813], [0.0, 0.0, 0.59375], [0.0, 0.0, 0.60938], [0.0, 0.0, 0.625], [0.0, 0.0, 0.64063], [0.0, 0.0, 0.65625], [0.0, 0.0, 0.67188], [0.0, 0.0, 0.6875], [0.0, 0.0, 0.70313], [0.0, 0.0, 0.71875], [0.0, 0.0, 0.73438], [0.0, 0.0, 0.75], [0.0, 0.0, 0.76563], [0.0, 0.0, 0.78125], [0.0, 0.0, 0.79688], [0.0, 0.0, 0.8125], [0.0, 0.0, 0.82813], [0.0, 0.0, 0.84375], [0.0, 0.0, 0.85938], [0.0, 0.0, 0.875], [0.0, 0.0, 0.89063], [0.0, 0.0, 0.90625], [0.0, 0.0, 0.92188], [0.0, 0.0, 0.9375], [0.0, 0.0, 0.95313], [0.0, 0.0, 0.96875], [0.0, 0.0, 0.98438], [0.0, 0.0, 1.0], [0.0, 0.015625, 1.0], [0.0, 0.03125, 1.0], [0.0, 0.046875, 1.0], [0.0, 0.0625, 1.0], [0.0, 0.078125, 1.0], [0.0, 0.09375, 1.0], [0.0, 0.10938, 1.0], [0.0, 0.125, 1.0], [0.0, 0.14063, 1.0], [0.0, 0.15625, 1.0], [0.0, 0.17188, 1.0], [0.0, 0.1875, 1.0], [0.0, 0.20313, 1.0], [0.0, 0.21875, 1.0], [0.0, 0.23438, 1.0], [0.0, 0.25, 1.0], [0.0, 0.26563, 1.0], [0.0, 0.28125, 1.0], [0.0, 0.29688, 1.0], [0.0, 0.3125, 1.0], [0.0, 0.32813, 1.0], [0.0, 0.34375, 1.0], [0.0, 0.35938, 1.0], [0.0, 0.375, 1.0], [0.0, 0.39063, 1.0], [0.0, 0.40625, 1.0], [0.0, 0.42188, 1.0], [0.0, 0.4375, 1.0], [0.0, 0.45313, 1.0], [0.0, 0.46875, 1.0], [0.0, 0.48438, 1.0], [0.0, 0.5, 1.0], [0.0, 0.51563, 1.0], [0.0, 0.53125, 1.0], [0.0, 0.54688, 1.0], [0.0, 0.5625, 1.0], [0.0, 0.57813, 1.0], [0.0, 0.59375, 1.0], [0.0, 0.60938, 1.0], [0.0, 0.625, 1.0], [0.0, 0.64063, 1.0], [0.0, 0.65625, 1.0], [0.0, 0.67188, 1.0], [0.0, 0.6875, 1.0], [0.0, 0.70313, 1.0], [0.0, 0.71875, 1.0], [0.0, 0.73438, 1.0], [0.0, 0.75, 1.0], [0.0, 0.76563, 1.0], [0.0, 0.78125, 1.0], [0.0, 0.79688, 1.0], [0.0, 0.8125, 1.0], [0.0, 0.82813, 1.0], [0.0, 0.84375, 1.0], [0.0, 0.85938, 1.0], [0.0, 0.875, 1.0], [0.0, 0.89063, 1.0], [0.0, 0.90625, 1.0], [0.0, 0.92188, 1.0], [0.0, 0.9375, 1.0], [0.0, 0.95313, 1.0], [0.0, 0.96875, 1.0], [0.0, 0.98438, 1.0], [0.0, 1.0, 1.0], [0.015625, 1.0, 0.98438], [0.03125, 1.0, 0.96875], [0.046875, 1.0, 0.95313], [0.0625, 1.0, 0.9375], [0.078125, 1.0, 0.92188], [0.09375, 1.0, 0.90625], [0.10938, 1.0, 0.89063], [0.125, 1.0, 0.875], [0.14063, 1.0, 0.85938], [0.15625, 1.0, 0.84375], [0.17188, 1.0, 0.82813], [0.1875, 1.0, 0.8125], [0.20313, 1.0, 0.79688], [0.21875, 1.0, 0.78125], [0.23438, 1.0, 0.76563], [0.25, 1.0, 0.75], [0.26563, 1.0, 0.73438], [0.28125, 1.0, 0.71875], [0.29688, 1.0, 0.70313], [0.3125, 1.0, 0.6875], [0.32813, 1.0, 0.67188], [0.34375, 1.0, 0.65625], [0.35938, 1.0, 0.64063], [0.375, 1.0, 0.625], [0.39063, 1.0, 0.60938], [0.40625, 1.0, 0.59375], [0.42188, 1.0, 0.57813], [0.4375, 1.0, 0.5625], [0.45313, 1.0, 0.54688], [0.46875, 1.0, 0.53125], [0.48438, 1.0, 0.51563], [0.5, 1.0, 0.5], [0.51563, 1.0, 0.48438], [0.53125, 1.0, 0.46875], [0.54688, 1.0, 0.45313], [0.5625, 1.0, 0.4375], [0.57813, 1.0, 0.42188], [0.59375, 1.0, 0.40625], [0.60938, 1.0, 0.39063], [0.625, 1.0, 0.375], [0.64063, 1.0, 0.35938], [0.65625, 1.0, 0.34375], [0.67188, 1.0, 0.32813], [0.6875, 1.0, 0.3125], [0.70313, 1.0, 0.29688], [0.71875, 1.0, 0.28125], [0.73438, 1.0, 0.26563], [0.75, 1.0, 0.25], [0.76563, 1.0, 0.23438], [0.78125, 1.0, 0.21875], [0.79688, 1.0, 0.20313], [0.8125, 1.0, 0.1875], [0.82813, 1.0, 0.17188], [0.84375, 1.0, 0.15625], [0.85938, 1.0, 0.14063], [0.875, 1.0, 0.125], [0.89063, 1.0, 0.10938], [0.90625, 1.0, 0.09375], [0.92188, 1.0, 0.078125], [0.9375, 1.0, 0.0625], [0.95313, 1.0, 0.046875], [0.96875, 1.0, 0.03125], [0.98438, 1.0, 0.015625], [1.0, 1.0, 0.0], [1.0, 0.98438, 0.0], [1.0, 0.96875, 0.0], [1.0, 0.95313, 0.0], [1.0, 0.9375, 0.0], [1.0, 0.92188, 0.0], [1.0, 0.90625, 0.0], [1.0, 0.89063, 0.0], [1.0, 0.875, 0.0], [1.0, 0.85938, 0.0], [1.0, 0.84375, 0.0], [1.0, 0.82813, 0.0], [1.0, 0.8125, 0.0], [1.0, 0.79688, 0.0], [1.0, 0.78125, 0.0], [1.0, 0.76563, 0.0], [1.0, 0.75, 0.0], [1.0, 0.73438, 0.0], [1.0, 0.71875, 0.0], [1.0, 0.70313, 0.0], [1.0, 0.6875, 0.0], [1.0, 0.67188, 0.0], [1.0, 0.65625, 0.0], [1.0, 0.64063, 0.0], [1.0, 0.625, 0.0], [1.0, 0.60938, 0.0], [1.0, 0.59375, 0.0], [1.0, 0.57813, 0.0], [1.0, 0.5625, 0.0], [1.0, 0.54688, 0.0], [1.0, 0.53125, 0.0], [1.0, 0.51563, 0.0], [1.0, 0.5, 0.0], [1.0, 0.48438, 0.0], [1.0, 0.46875, 0.0], [1.0, 0.45313, 0.0], [1.0, 0.4375, 0.0], [1.0, 0.42188, 0.0], [1.0, 0.40625, 0.0], [1.0, 0.39063, 0.0], [1.0, 0.375, 0.0], [1.0, 0.35938, 0.0], [1.0, 0.34375, 0.0], [1.0, 0.32813, 0.0], [1.0, 0.3125, 0.0], [1.0, 0.29688, 0.0], [1.0, 0.28125, 0.0], [1.0, 0.26563, 0.0], [1.0, 0.25, 0.0], [1.0, 0.23438, 0.0], [1.0, 0.21875, 0.0], [1.0, 0.20313, 0.0], [1.0, 0.1875, 0.0], [1.0, 0.17188, 0.0], [1.0, 0.15625, 0.0], [1.0, 0.14063, 0.0], [1.0, 0.125, 0.0], [1.0, 0.10938, 0.0], [1.0, 0.09375, 0.0], [1.0, 0.078125, 0.0], [1.0, 0.0625, 0.0], [1.0, 0.046875, 0.0], [1.0, 0.03125, 0.0], [1.0, 0.015625, 0.0], [1.0, 0.0, 0.0], [0.98438, 0.0, 0.0], [0.96875, 0.0, 0.0], [0.95313, 0.0, 0.0], [0.9375, 0.0, 0.0], [0.92188, 0.0, 0.0], [0.90625, 0.0, 0.0], [0.89063, 0.0, 0.0], [0.875, 0.0, 0.0], [0.85938, 0.0, 0.0], [0.84375, 0.0, 0.0], [0.82813, 0.0, 0.0], [0.8125, 0.0, 0.0], [0.79688, 0.0, 0.0], [0.78125, 0.0, 0.0], [0.76563, 0.0, 0.0], [0.75, 0.0, 0.0], [0.73438, 0.0, 0.0], [0.71875, 0.0, 0.0], [0.70313, 0.0, 0.0], [0.6875, 0.0, 0.0], [0.67188, 0.0, 0.0], [0.65625, 0.0, 0.0], [0.64063, 0.0, 0.0], [0.625, 0.0, 0.0], [0.60938, 0.0, 0.0], [0.59375, 0.0, 0.0], [0.57813, 0.0, 0.0], [0.5625, 0.0, 0.0], [0.54688, 0.0, 0.0], [0.53125, 0.0, 0.0], [0.51563, 0.0, 0.0], [0.5, 0.0, 0.0]])

def gray2jet(img, normalize=True):
    assert img.ndim == 2
    m, n = img.shape
    jet = np.zeros((m, n, 3))

    if normalize:
        largest = np.max(img)
        if largest != 0:
            img /= largest
            img *= 255
        else:
            img = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            val = int(round(img[i][j]))
            if val < 0 or val > 255:
                continue
            jet[i][j][:] = jetMap[val][:]

    return jet

def read_image(path):
    file = open(path, 'r')
    content = file.read()
    content = content.split(',')
    content = np.array([float(k) for k in content])

    img = np.reshape(content,(vgg_size, vgg_size)) if args.gray else np.reshape(content,(vgg_size, vgg_size, 3))
    return img


def run(read_path, model, write_path, start):
    assert os.path.exists(read_path)

    print(read_path, 'start!!!')

    file_list = os.listdir(read_path)

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    count = -1
    beginTime = time.time()
    for file in file_list:
        count += 1
        if count < start:
            continue
        if count % 100 == 0:
            print(count, time.time() - beginTime)

        file_to_write = open(write_path + file, 'w')

        img = read_image(read_path + file)
        img = gray2jet(img) if args.gray else img

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        block5_pool_features = model.predict(x)

        sp = block5_pool_features.shape

        y = np.zeros((sp[2], sp[3]))
        for j in range(sp[2]):
            tmp = np.reshape(block5_pool_features[0][:][j][:], (sp[1], sp[3]))
            for p in range(sp[1]):
                for q in range(sp[3]):
                    tmp[p][q] = max(0, tmp[p][q])
            y[j] = sum(tmp) / (sp[1] * 1.0)
        y = sum(y) / (sp[2] * 1.0)

        file_to_write.write(','.join(str(i) for i in y.tolist()))
        file_to_write.close()

    print(read_path, 'end!!!')

if __name__ == '__main__':
    base_model = VGG19(weights='imagenet', include_top=args.top)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(args.layer).output)
    if status1:
        run(x_path, model, x_path_w, start1)
    if status2:
        run(y_path, model, y_path_w, start2)
    if status3:
        run(x_path_mc, model, x_path_mc_w, start3)
    if status4:
        run(y_path_mc, model, y_path_mc_w, start4)



