from scipy.interpolate import lagrange
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_data(fn):
    file = open(fn, 'r')
    content = file.read()
    content = content.split('\n')
    tlen = len(content)
    if tlen == 1:
        content = [float(i) for i in content[0].split(',')]
        return np.reshape(np.array(content), (len(content) // 18, 18))
    else:
        while(content[len(content)-1] == ""):
            content.pop()
        for i in range(len(content)):
            content[i] = [float(k) for k in content[i].split(',')]
        return np.array(content)

def get_xy(matrix, key):
    w = matrix.shape[0]
    x = []
    y = []
    interest = []
    for i in range(w):
        if matrix[i,key] == 0:
            interest.append(i)
        else:
            x.append(i)
            y.append(matrix[i, key])
    return x, y, interest, len(interest)/w

def interp_lagrange(x, y, xx):
    # 调用拉格朗日插值，得到插值函数p
    p = lagrange(x, y)
    yy = p(xx)
    plt.plot(x, y, "b*")
    plt.plot(xx, yy, "ro")
    plt.show()

def interp(x, y, xx, kind="linear"):#'nearest', 'zero', 'linear', 'quadratic'
    plt.plot(x, y, 'b*')
    while(xx[len(xx)-1] > x[len(x)-1]):
        xx.pop()
    while(xx[0] < x[0]):
        xx.pop(0)
    print(xx)
    f = interpolate.interp1d(x, y, kind=kind)
    yy = f(xx)  # 计算插值结果
    plt.plot(xx, yy, "ro")
    allx = [i for i in range(x[0],x[len(x)-1])]
    plt.plot(allx, f(allx), "g")
    plt.show()


def fit(x, y, xx):
    plt.plot(x, y, 'b*')
    f = np.polyfit(x, y, 2)
    p = np.poly1d(f)
    yy = p(xx)  # 计算插值结果
    plt.plot(xx, yy, "ro")
    plt.plot(x, p(x), "g")
    plt.show()
import datetime, time
if __name__ == '__main__':
    # NUMBER = 20
    # eps = np.random.rand(NUMBER) * 2
    #
    # x_path = "D:/graduation_project/workspace/dataset/HMDB51/ori_x/"
    # y_path = "D:/graduation_project/workspace/dataset/HMDB51/ori_x/"
    # fn = "0_1.txt"
    # x = load_data(x_path + fn)
    # y = load_data(y_path + fn)
    # tx, ty, txx, rho = get_xy(x, 16)
    # print(rho)
    # print(tx)
    # print(ty)
    # print(txx)
    # # fit(tx, ty, txx)
    # interp(tx, ty, txx)
    # while True:
    #     current_time = time.localtime(time.time())
    #     print(current_time.tm_hour, current_time.tm_min)
    #     if current_time.tm_hour >= 2 and current_time.tm_min >= 0:
    #         break
    #     time.sleep(1)
    #
    # while True:
    #     now = datetime.datetime.now()
    #     print(now.hour, now.minute)
    #     time.sleep(1)

    for i in range(64, 1025, 64):
        print(i)
