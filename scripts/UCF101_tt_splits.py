import shutil
import os
###############
num = 3
generate_train = True
generate_train_labels = False
generate_test = False
generate_test_labels = False
##############

source_folder = 'D:/graduation_project/workspace/dataset/UCF101/'
target_folder_root = 'D:/graduation_project/workspace/dataset/UCF101_train_test_splits/'

trainlist_name = 'trainlist0'+str(num)+'.txt'
testlist_name = 'testlist0'+str(num)+'.txt'

train_target_folder = target_folder_root + 'train' + str(num) + '/'
test_target_folder = target_folder_root + 'test' + str(num) + '/'

if generate_train:
    name = []
    kind = []
    with open(source_folder+trainlist_name, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
            name_tmp, kind_tmp = [x for x in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            name.append(name_tmp)  # 添加新读取的数据
            kind.append(int(kind_tmp))
        file_to_read.close()

    print(kind)
    print(name)
    count = 0
    for i in name:
        new_name = str(count)+'_'+str(kind[count])+'.avi'
        shutil.copy(source_folder+i, train_target_folder+new_name)
        count = count + 1
        if count % 100 == 0:
            print(count)

if generate_train_labels:
    kind = []
    with open(source_folder + trainlist_name, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            name_tmp, kind_tmp = [x for x in lines.split()]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            kind.append(int(kind_tmp))
        file_to_read.close()

    print(kind)
    file_to_write = open(target_folder_root + 'train' + str(num) + '_labels.txt', 'w')
    file_to_write.write(' '.join([str(i) for i in kind]))
    file_to_write.close()

if generate_test:
    rec = []
    name2type = {}

    with open(target_folder_root+'classInd.txt', 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
            tmp = lines.split(' ')
            name2type[tmp[1].split('\n')[0]] = tmp[0]
        file_to_read.close()
        print(name2type)

    with open(source_folder+testlist_name, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
            rec.append(lines.split('\n')[0])
        file_to_read.close()
        print(rec)

    count = 0
    for i in rec:
        tmp = i.split('_')[1]
        new_name = str(count) + '_' + str(name2type[tmp]) + '.avi'
        print(new_name)
        shutil.copy(source_folder+i, test_target_folder+new_name)
        count = count + 1