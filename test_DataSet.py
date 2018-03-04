import cv2
from scripts.DataSet import *
import numpy
import random
import time
import profile
import matplotlib.pyplot as plt # plt 用于显示图片
# seed = None
# seed1, seed2 = random_seed.get_seed(seed)
# print(seed1, seed2)
# numpy.random.seed(seed1 if seed is None else seed2)

begin_time = time.time()
dummy = []
for i in range(20):
    for j in range(25):
        img = cv2.imread('D:/graduation_project/workspace/dataset/UCF101_train_test_splits/train1_spatial/0_0_1.jpg')
        dummy.append(img)
print(time.time()-begin_time, 's')
# tmp2 = [[[0 for x in range(239)] for y in range(319)] for z in range(3)]
# test = []
# test.append(img)
# test.append(img)
# test.append(tmp2)
# tmp = numpy.array(test)
# print(tmp.shape)


arr = []
for i in range(2):
    arr.append(numpy.arange(10))
#
# for i in range(10):
#     numpy.random.shuffle(arr[0])
#     numpy.random.shuffle(arr[1])
#     print(arr)


train = DataSet('UCF', 'train1', 25)
begin_time = time.time()
images, labels = train.next_batch(2)
print(time.time() - begin_time, 's')
print(images.shape)
print(labels.shape)
print(labels[0])
plt.imshow(images[0])
plt.show()
def next_batch_test():
    train = DataSet('UCF', 'train1', 25)
    images, labels = train.next_batch(500)

def next_batch_test2():
    train = DataSet('UCF', 'train1', 25)

# profile.run("next_batch_test2()")
# profile.run("next_batch_test()")


# for i in range(480):
#     res = train.next_batch(500)
#     print(len(res),str(train.index_in_epoch_video), train.index_frame)
#
# for i in range(10):
#     print(random.randint(0, 1))