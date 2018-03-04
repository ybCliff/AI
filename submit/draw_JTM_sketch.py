import numpy as np
import os, cv2
import random, time
from matplotlib import pyplot as plt
root = "D:/graduation_project/workspace/dataset/HMDB51/"
avg_split_num = 11
loop_num = 5

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

def draw(x, y):
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    fig = plt.figure()

    #################################################################
    canvas = np.zeros((50, 349, 3), np.uint8)
    canvas[canvas == 0] = 255
    # print(canvas)
    w = x.shape[0]
    for i in range(w - 1):
        cv2.line(canvas, (x[i], y[i]), (x[i + 1], y[i + 1]), (0, 0, 0), 3)
        cv2.line(canvas, (x[i], y[i] - 4), (x[i], y[i] + 4), (0, 0, 0), 2)
    print(x[w - 1], y[w - 1])
    cv2.line(canvas, (x[w - 1], y[w - 1] - 4), (x[w - 1], y[w - 1] + 4), (0, 0, 0), 2)

    ax = fig.add_subplot(211)
    ax.set_xlim([1, 360])
    ax.set_ylim([1, 60])
    plt.imshow(canvas)
    ax.text(x[0] - 10, y[0] + 10, r"$ JTM_0 $", fontsize=10)
    ax.text(x[1] - 10, y[1] + 10, r"$ JTM_1 $", fontsize=10)
    ax.text(x[2] - 10, y[2] + 10, r"$ JTM_2 $", fontsize=10)
    ax.text(x[int((w + 1) / 2)] - 10, y[int((w + 1) / 2)] + 13, r"$......$", fontsize=10)
    ax.text(x[w - 1] - 10, y[w - 1] + 10, r"$ JTM_{n-1} $", fontsize=10)

    ax.text(x[0] + 4, y[0] - 13, r"$f(1)$", fontsize=10)
    ax.text(x[1] + 4, y[1] - 13, r"$f(2)$", fontsize=10)
    ax.text(x[w - 2] + 1, y[w - 2] - 13, r"$f(n-1)$", fontsize=10)

    ax.set_title("$f(i)=\{t_1^i, t_2^i, ..., t_m^i\}$")
    ax.set_xlabel("(a)")

    #################################################################

    canvas = np.zeros((50, 349, 3), np.uint8)
    canvas[canvas==0] = 255
    # print(canvas)
    w = x.shape[0]
    for i in range(w-1):
        cv2.line(canvas, (x[i], y[i]), (x[i+1], y[i+1]), jetMap[i].tolist(), 3)
        cv2.line(canvas, (x[i], y[i]-4), (x[i], y[i]+4), (0,0,0), 2)
    print(x[w-1], y[w-1])
    cv2.line(canvas, (x[w-1], y[w-1] - 4), (x[w-1], y[w-1] + 4), (0, 0, 0), 2)

    ax = fig.add_subplot(212)
    ax.set_xlim([1, 360])
    ax.set_ylim([1, 60])
    plt.imshow(canvas)
    ax.text(x[0] - 10, y[0] + 10, r"$ JTM_0 $", fontsize=10)
    ax.text(x[1] - 10, y[1] + 10, r"$ JTM_1 $", fontsize=10)
    ax.text(x[2] - 10, y[2] + 10, r"$ JTM_2 $", fontsize=10)
    ax.text(x[int((w+1)/2)]-10, y[int((w+1)/2)] + 13, r"$......$", fontsize=10)
    ax.text(x[w-1] - 10, y[w-1] + 10, r"$ JTM_{n-1} $", fontsize=10)

    ax.text(x[0] + 4, y[0] - 13, r"$f(1)$", fontsize=10)
    ax.text(x[1] + 4, y[1] - 13, r"$f(2)$", fontsize=10)
    ax.text(x[w-2] + 1, y[w - 2] - 13, r"$f(n-1)$", fontsize=10)

    ax.set_title("$f(i)=\{C\_t_1^i, C\_t_2^i, ..., C\_t_m^i\}$")
    ax.set_xlabel("(b)")

    # plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    x = np.linspace(30, 330, 11)
    y = np.linspace(25, 25, 11)
    draw(x, y)




