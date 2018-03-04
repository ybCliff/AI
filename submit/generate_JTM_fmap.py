from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import argparse
import numpy as np
import os, time, cv2

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--model', type=str, default='vgg19')
parser.add_argument('--layer', type=str, default='block5_pool')
parser.add_argument('--fusion', type=str, default='mean')   # mean  or  max
parser.add_argument('--folder', type=str, default='JTM_mc/15')   # JTM_mc/x, JTM_ori/x, JDM_mc/x, JDM_ori/x, spatial_x/frame
parser.add_argument('--top', type=bool, default=False)
args = parser.parse_args()

root = "D:/graduation_project/workspace/dataset/"+args.dataset+'/'
vgg_size = 224
folder_name = args.model + '_' + args.layer + '_' + args.fusion


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
        if count < start or '.jpg' not in file:
            continue
        if count % 100 == 0:
            print(count, time.time() - beginTime)

        txt_name = file.split('.')[0] + '.txt'
        file_to_write = open(write_path + txt_name, 'w')

        img = cv2.imread(read_path + file)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)
        if args.layer == 'block5_pool':
            y = np.mean(features, axis=0)
            if args.fusion == 'mean':
                y = np.mean(y, axis=0)
                y = np.mean(y, axis=0)
            else:
                y = np.max(y, axis=0)
                y = np.max(y, axis=0)
        else:
            y = features

        file_to_write.write(','.join(str(i) for i in y.tolist()))
        file_to_write.close()

    print(read_path, 'end!!!')

if __name__ == '__main__':
    # while True:
    #     current_time = time.localtime(time.time())
    #     if current_time.tm_hour >= 2 and current_time.tm_min >= 15:
    #         break
    #     time.sleep(10)

    base_model = VGG19(weights='imagenet', include_top=args.top)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(args.layer).output)

    read_path1 = root + 'train1/' + args.folder + '/'
    write_fmap_path1 = read_path1 + folder_name + '/'

    read_path2 = root + 'test1/' + args.folder + '/'
    write_fmap_path2 = read_path2 + folder_name + '/'

    run(read_path1, model, write_fmap_path1, 0)
    run(read_path2, model, write_fmap_path2, 0)




