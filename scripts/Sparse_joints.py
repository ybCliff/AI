import os
import numpy as np
x_path = 'D:/graduation_project/workspace/dataset/HMDB51/ori_x/'
y_path = 'D:/graduation_project/workspace/dataset/HMDB51/ori_y/'

def sparse(matrix):
    w, h = matrix.shape
    not_zero_count = 0
    for i in range(w):
        for j in range(h):
            if matrix[i, j] != 0:
                not_zero_count += 1
    return not_zero_count * 1.0 / (w * h)

def load_data(fn):
    file_to_read = open(fn, 'r')
    content = file_to_read.read()
    content = content.split('\n')
    while content[len(content)-1] == '':
        content.pop()
    for i in range(len(content)):
        content[i] = [float(k) for k in content[i].split(',')]
    return content



def joints_sparse(matrix):
    w, h = matrix.shape
    count = np.zeros((18))
    for i in range(w):
        for j in range(h):
            if matrix[i, j] != 0:
                count[j] += 1.0
    return count / w

def get_sparse(path):
    x_sparse = [0.0] * 51
    x_count = [0] * 51
    y_sparse = [0.0] * 51
    y_count = [0] * 51
    res_x = [0.0] * 51
    res_y = [0.0] * 51
    joints_sparse_rec = np.zeros((51, 18))

    file_list = os.listdir(path)

    for filename in file_list:
        x_matrix = load_data(x_path + filename)
        y_matrix = load_data(y_path + filename)

        type = int(filename.split('.')[0].split('_')[1]) - 1
        xsp = sparse(np.array(x_matrix))
        ysp = sparse(np.array(y_matrix))
        x_sparse[type] += xsp
        y_sparse[type] += ysp
        x_count[type] += 1
        y_count[type] += 1

        tmp = joints_sparse(np.array(x_matrix))
        joints_sparse_rec[type] += tmp

    for i in range(51):
        res_x[i] = x_sparse[i] / x_count[i]
        res_y[i] = y_sparse[i] / y_count[i]
        joints_sparse_rec[i] /= x_count[i]


    return res_x, res_y, joints_sparse_rec

read_path = "D:/graduation_project/workspace/dataset/HMDB51/ori/"
save = '../evaluation_statistics/sparse_joints/'
if not os.path.exists(save):
    os.makedirs(save)

tx, ty, tj = get_sparse(read_path)

file_to_write = open(save+'all_mc_linear.csv', 'w')
print(','.join([str(i) for i in range(51)]), file=file_to_write)
res = [0.0] * 51
for i in range(51):
    res[i] = 0.5 * (tx[i] + ty[i])
    print(i, res[i])
print(','.join([str(i) for i in res]), file=file_to_write)
file_to_write.close()

file_to_write = open(save+'all_joints_rec.csv', 'w')
print(','.join([str(i) for i in range(-1, 18)]), file=file_to_write)
for i in range(51):
    print(str(i) + ',', end="", file=file_to_write)
    print(','.join([str(i) for i in tj[i]]), file=file_to_write)
file_to_write.close()

# load_data("D:/graduation_project/workspace/dataset/HMDB51/ori_x/0_1.txt")