from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import argparse
import numpy as np
import os, time, cv2
parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--scope', type=str, default='train1')
parser.add_argument('--layer', type=str, default='block5_pool')
parser.add_argument('--top', type=bool, default=True)
parser.add_argument('--gray', type=bool, default=False)
parser.add_argument('--ten', type=bool, default=True)
args = parser.parse_args()
root = "D:/graduation_project/workspace/dataset/"
spatial_root = root + args.dataset + '_train_test_splits/' + args.scope + '_spatial/'
spatial_write = spatial_root + args.layer + '/'

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}

file_finally_num = Dict[args.dataset][args.scope]

if not os.path.exists(spatial_write):
    os.makedirs(spatial_write)

status1 = True

vgg_size = 224


def get_ten(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (max([h, vgg_size]), max([w, vgg_size])))
    h, w = img.shape[:2]
    lu = img[:vgg_size, :vgg_size]
    lb = img[h-vgg_size:, :vgg_size]
    ru = img[:vgg_size, w-vgg_size:]
    rb = img[h-vgg_size:, w-vgg_size:]

    cid_h = int((h-vgg_size)/2.0)
    cid_w = int((w-vgg_size)/2.0)
    center = img[cid_h:cid_h+vgg_size, cid_w:cid_w+vgg_size]

    return [lu, lb, ru, rb, center, lu[:, ::-1], lb[:, ::-1], ru[:, ::-1], rb[:, ::-1], center[:, ::-1]]



def run(read_path, model, write_path, start):
    assert os.path.exists(read_path)

    print(read_path, 'start!!!')

    file_list = os.listdir(read_path)

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    count = -1
    beginTime = time.time()
    for file in file_list:
        img = cv2.imread(read_path + file)
        if img is None:
            continue

        if args.ten:
            img = np.array(get_ten(img))
        else:
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
        count += 1
        if count < start:
            continue
        if count % 100 == 0:
            print(count, time.time() - beginTime)
        video_num = file.split('.')[0].split('_')[0]
        type = file.split('.')[0].split('_')[2]
        tmp_scope = '_1x512/'
        tmp_path = tmp_scope + video_num + '_' + type + '/'
        fn = str(count) + '_' + type + '.txt'
        if not os.path.exists(write_path + tmp_path):
            os.makedirs(write_path + tmp_path)
        file_to_write = open(write_path + tmp_path + fn, 'w')
        file_to_write2 = open(write_path + tmp_scope + fn, 'w')

        x = np.zeros(img.shape)
        for i in range(img.shape[0]):
            x[i] = image.img_to_array(img[i])

        x = preprocess_input(x)

        features = model.predict(x)

        if args.layer == 'block5_pool':
            y = np.mean(features, axis=0)
            y = np.mean(y, axis=0)
            y = np.mean(y, axis=0)
        else:
            y = features
        file_to_write.write(','.join(str(i) for i in y.tolist()))
        file_to_write2.write(','.join(str(i) for i in y.tolist()))
        file_to_write.close()
        file_to_write2.close()
    print(read_path, 'end!!!')

if __name__ == '__main__':
    base_model = VGG19(weights='imagenet', include_top=args.top)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(args.layer).output)
    # if status1:
    #     run(spatial_root, model, spatial_write, 0)
    spatial_root = root + args.dataset + '_train_test_splits/train1_spatial/'
    spatial_write = spatial_root + args.layer + '/ten' + str(int(args.ten)) + '/'
    spatial_root2 = root + args.dataset + '_train_test_splits/test1_spatial/'
    spatial_write2 = spatial_root2 + args.layer + '/ten' + str(int(args.ten)) + '/'

    spatial_root3 = root + args.dataset + '_train_test_splits/train2_spatial/'
    spatial_write3 = spatial_root3 + args.layer + '/ten' + str(int(args.ten)) + '/'
    spatial_root4 = root + args.dataset + '_train_test_splits/test2_spatial/'
    spatial_write4 = spatial_root4 + args.layer + '/ten' + str(int(args.ten)) + '/'

    spatial_root5 = root + args.dataset + '_train_test_splits/train3_spatial/'
    spatial_write5 = spatial_root5 + args.layer + '/ten' + str(int(args.ten)) + '/'
    spatial_root6 = root + args.dataset + '_train_test_splits/test3_spatial/'
    spatial_write6 = spatial_root6 + args.layer + '/ten' + str(int(args.ten)) + '/'

    run(spatial_root, model, spatial_write, 0)
    run(spatial_root2, model, spatial_write2, 0)
    # run(spatial_root3, model, spatial_write3, 0)
    # run(spatial_root4, model, spatial_write4, 0)
    # run(spatial_root5, model, spatial_write5, 0)
    # run(spatial_root6, model, spatial_write6, 0)




