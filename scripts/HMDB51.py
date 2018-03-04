import shutil
import os
video_source_path = "D:/graduation_project/workspace/dataset/HMDB51/"
txt_source_path = "D:/graduation_project/workspace/dataset/HMDB51_tt_txt/"
target_root_path = "D:/graduation_project/workspace/dataset/HMDB51/video/"

if not os.path.exists(target_root_path):
    os.makedirs(target_root_path)

file_list = os.listdir(txt_source_path)
type = 0
Dict = {}
tt_count = {"train1": 0, "test1": 0, "train2": 0, "test2": 0, "train3": 0, "test3": 0}
vDict = {}
vDict2 = {}
vCount = -1
split2num = {"train1":0, "train2":1, "train3":2, "test1":3, "test2":4, "test3":5}
for file in file_list:
    tmp = file.split("_test_")
    tmp_video_path = video_source_path + tmp[0] + '/'

    if tmp[0] not in Dict.keys():
        type += 1
        Dict[tmp[0]] = type

    type_num = Dict[tmp[0]]
    num = tmp[1][5]

    file_to_read = open(txt_source_path + file, 'r')

    content = file_to_read.readline()
    while content:
        content = content.split(' ')
        if content[0] not in vDict.keys():
            vCount += 1
            name = str(vCount) + '_' + str(type_num) + '.avi'
            vDict[content[0]] = name
            vDict2[name] = []
            video_path = tmp_video_path + content[0]
            target_path = target_root_path + name
            shutil.copy(video_path, target_path)

        n = vDict[content[0]]
        if content[1] is '1' or '2':
            split = "train" + num if content[1] is '1' else "test" + num
            target_name = str(tt_count[split]) + '_' + str(type_num) + '_' + str(split2num[split])
            vDict2[n].append(target_name)
            tt_count[split] += 1
        content = file_to_read.readline()

    file_to_read.close()

# Dict = sorted(Dict.items(), key=lambda d: d[1])
# file_to_write = open(target_root_path + 'map.txt', 'w')
# for k in Dict.keys():
#     file_to_write.write(k + ' ' + str(Dict[k]) + '\n')
# file_to_write.close()


file_to_write = open(video_source_path + 'vDict.txt', 'w')
for k in vDict.keys():
    print(k, vDict[k], file=file_to_write)
file_to_write.close()
file_to_write = open(video_source_path + 'vDict2.txt', 'w')
for k in vDict2.keys():
    print(k, ','.join([i for i in vDict2[k]]), file=file_to_write)
file_to_write.close()

# Dict = sorted(Dict.items(), key=lambda e: e[1], reverse=False)
file_to_write = open(video_source_path + 'map.txt', 'w')
for k in Dict.keys():
    print(k, str(Dict[k]), file=file_to_write)
file_to_write.close()