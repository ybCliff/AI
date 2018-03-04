import os
import shutil

path = 'D:/graduation_project/workspace/dataset/HMDB51_train_test_splits/test1_spatial/block5_pool/_1x512/'

mylist = os.listdir(path)
for file in mylist:
    if '.txt' in file:
        continue
    type = file.split('_')[1]
    new_path = path + file + '/'
    file_list = os.listdir(new_path)
    for key in file_list:
        pre = key.split('.')[0]
        os.rename(new_path+key, new_path+pre+'_'+type+'.txt')
    file_list = os.listdir(new_path)
    for key in file_list:
        shutil.copy(new_path+key, path+key)