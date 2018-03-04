import shutil
import os
root = "D:/graduation_project/workspace/dataset/HMDB51/"
scopex = 'ori_x_mc/'
scopey = 'ori_y_mc/'
x_path = root + scopex
y_path = root + scopey
num2split = {0:"train1", 1:"train2", 2:"train3", 3:"test1", 4:"test2", 5:"test3"}

def run(split):
    video_path = root + split + '/video/'
    file_list = os.listdir(video_path)
    trg_x_path = root + split + '/' + scopex
    trg_y_path = root + split + '/' + scopey
    if not os.path.exists(trg_x_path):
        os.makedirs(trg_x_path)
    if not os.path.exists(trg_y_path):
        os.makedirs(trg_y_path)
    for file in file_list:
        txt_name = file.split('.')[0] + '.txt'
        shutil.copy(x_path+txt_name, trg_x_path+txt_name)
        shutil.copy(y_path+txt_name, trg_y_path+txt_name)



for i in range(6):
    split = num2split[i]
    run(split)