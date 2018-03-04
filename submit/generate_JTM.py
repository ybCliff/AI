import numpy as np
import os, cv2
import random, time
from matplotlib import pyplot as plt
root = "D:/graduation_project/workspace/dataset/HMDB51/"
avg_split_num = 11
loop_num = 20

jetMap = np.array([
[0,0,0.66667],
[0,0,1],
[0,0.33333,1],
[0,0.66667,1],
[0,1,1],
[0.33333,1,0.66667],
[0.66667,1,0.33333],
[1,1,0],
[1,0.66667,0],
[1,0.33333,0]
])
jetMap *= 255
jetMap = jetMap.astype(np.int64)
print(jetMap)
read_video_path = root + 'video/'

def load_data(fn, joints_keys=18):
    file = open(fn, 'r')
    content = file.read()
    content = content.split('\n')
    tlen = len(content)
    if tlen == 1:
        content = [float(i) for i in content[0].split(',')]
        return np.reshape(np.array(content), (len(content) // joints_keys, joints_keys))
    else:
        while(content[len(content)-1] == ""):
            content.pop()
        for i in range(len(content)):
            content[i] = [float(k) for k in content[i].split(',')]
        return np.array(content)

def draw(x, y, read_video_name, op=-1):
    vc = cv2.VideoCapture(read_video_path + read_video_name)  # 读入视频文件
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    vc.release()
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[canvas==0] = 255
    # print(canvas)
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    w = x.shape[0]
    if op != -1:
        for i in range(w-1):
            canvas = cv2.line(canvas, (x[i, op], y[i, op]), (x[i+1, op], y[i+1, op]), jetMap[w-i-2].tolist(), 3)
    else:
        for j in range(18):
            for i in range(w-1):
                if x[i + 1, j] == 0 and y[i + 1, j] == 0:
                    continue
                if x[i, j] == 0 and y[i, j] == 0:
                    continue
                canvas = cv2.line(canvas, (x[i, j], y[i, j]), (x[i + 1, j], y[i + 1, j]),jetMap[w - i - 2].tolist(), 3)
    return canvas


def display(x, y, id, type, JTM_path, read_video_name):
    w = x.shape[0]
    arr = np.array([i for i in range(w)])
    split = np.array_split(arr, avg_split_num)
    for k in range  (loop_num):
        lst = []
        for i in range(avg_split_num):
            lst.append(random.sample(split[i].tolist(), 1)[0])

        img = draw(x[lst], y[lst], read_video_name)
        img_name = id + '_' + str(k) + '_' + type + '.jpg'
        # print(img_name)
        cv2.imwrite(JTM_path+img_name,img)


def run(scope):
    x_path = root + scope + '/ori_x_mc/'
    y_path = root + scope + '/ori_y_mc/'
    JTM_path = root + scope + '/JTM_mc/' + str(loop_num) + '/'
    if not os.path.exists(JTM_path):
        os.makedirs(JTM_path)
    filelist = os.listdir(x_path)
    count = 0
    beginTime = time.time()
    for fn in filelist:
        count += 1
        if count % 50 == 0:
            print(count, time.time() - beginTime)
        read_video_name = fn.split('.')[0] + '.avi'
        tmp = fn.split('.')[0].split('_')
        x = load_data(x_path + fn)
        y = load_data(y_path + fn)
        display(x, y, tmp[0], tmp[1], JTM_path, read_video_name)

run("train1")
run("test1")
