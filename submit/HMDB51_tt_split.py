import shutil
import os
root = "D:/graduation_project/workspace/dataset/HMDB51/"
video_path = root + 'video/'
txt_path = root + "vDict2.txt"

num2split = {0:"train1", 1:"train2", 2:"train3", 3:"test1", 4:"test2", 5:"test3"}

#要省略的类别——不会出现在训练、测试集中
#omit[i] = type - 1
omit = [1, 3, 8, 10,
        11, 12, 14, 19,
        24, 27,
        32, 36, 38,
        40, 41, 42, 43, 47, 48, 49, 50]



file_to_read = open(txt_path, 'r')
content = file_to_read.read()
content = content.split('\n')

for i in range(len(content)):
    tmp = content[i].split(' ')
    trg_video_name = tmp[0]

    tmp2 = tmp[1].split(',')
    for j in range(len(tmp2)):
        tmp3 = tmp2[j]
        a, b, num = tmp3.split('_')
        split = num2split[int(num)]
        trg_path = root + split + '/video/'
        if not os.path.exists(trg_path):
            os.makedirs(trg_path)
        if (int(b) - 1) in omit:
            continue
        shutil.copy(video_path+trg_video_name, trg_path+trg_video_name)
