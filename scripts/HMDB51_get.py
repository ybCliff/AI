import shutil
import os
txt_path = "D:/graduation_project/workspace/dataset/HMDB51/vDict2.txt"
trg_path = "D:/graduation_project/workspace/dataset/HMDB51/ori/"
root = "D:/graduation_project/workspace/dataset/HMDB51_train_test_splits/"
if not os.path.exists(trg_path):
    os.makedirs(trg_path)

num2split = {0:"train1", 1:"train2", 2:"train3", 3:"test1", 4:"test2", 5:"test3"}

file_to_read = open(txt_path, 'r')
content = file_to_read.read()
content = content.split('\n')

for i in range(len(content)):
    tmp = content[i].split(' ')
    trg_txt_name = tmp[0].split('.')[0] + '.txt'

    tmp2 = tmp[1].split(',')[0]
    a, b, num = tmp2.split('_')
    split = num2split[int(num)]
    sor_txt_path = root + split + '_joints/ori/'
    sor_txt_name = a + '_' + b + '.txt'
    shutil.copy(sor_txt_path+sor_txt_name, trg_path+trg_txt_name)


