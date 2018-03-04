import os,cv2
import argparse, time
import numpy as np
parser = argparse.ArgumentParser(description='')
parser.add_argument('--split', type=str, default='1')
args = parser.parse_args()

root = "D:/graduation_project/workspace/dataset/HMDB51/"
train = "train"+args.split
test = "test"+args.split
train_path = root + train + '/JTM/'
test_path = root + test + '/JTM/'

def load_img(path):
    file_list = os.listdir(path)
    res = []
    count = 0
    beginTime = time.time()
    for file in file_list:
        count += 1
        if count % 100 == 0:
            print(count, time.time() - beginTime)
        img = cv2.imread(path + file)
        res.append(img)
    return np.array(res)

my_train = load_img(train_path)
my_test = load_img(test_path)
