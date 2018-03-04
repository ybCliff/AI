#coding:utf-8
'''
线性层的softmax回归模型识别手写字
'''
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
root = "D:/graduation_project/workspace/dataset/UCF101_train_test_splits/train1_feature_files/vgg16_fc7/"


def read_data(file_num):
    pre = file_num * 10000
    post = pre + 9999
    x = []
    y = []
    file_name = str(pre) + '_' + str(post) + '.txt'
    file = open(root+file_name, 'r')
    context = file.readline()
    tmp = context.split(' ')
    # print(tmp[0])
    # print(tmp[1])
    # print(tmp[0].split(','))
    while context:
        tmp = context.split(' ')
        # print(tmp[0])
        # print(tmp[1])
        arr = tmp[0].split(',')
        tmp_y = 101 * [0.0]
        tmp_y[int(tmp[1]) - 1] = 1.0
        y.append(tmp_y)
        x.append([float(i) for i in arr])
        context = file.readline()
    file.close()
    x = np.array(x)
    y = np.array(y)
    return x, y

X, Y = read_data(0)
# X = X[0:1000, :]
# Y = Y[0:1000, :]
x = tf.placeholder("float", [None, 4096])
# w = tf.Variable(tf.truncated_normal([4096,101]))
# b = tf.Variable(tf.truncated_normal([101]))
w = tf.Variable(tf.zeros([4096,101]), 'weight')
b = tf.Variable(tf.zeros([101]), 'bias')
y = tf.nn.softmax(tf.matmul(x,w) + b)

# loss
y_ = tf.placeholder("float", [None, 101])
# loss1 = slim.losses.softmax_cross_entropy(y, y_)
# loss2 = tf.reduce_mean(tf.matmul(w, w, transpose_b=True))
# loss3 = tf.reduce_mean(b * b)
# loss = loss1 + loss2 + loss3

loss = slim.losses.softmax_cross_entropy(y, y_)
# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化
init = tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    # 迭代

    for i in range(2000000):
        tmp = y.eval(feed_dict={x: X})
        # print(tmp[0, 0:3])
        # print(w.eval()[0, 0:3])
        # print(b.eval()[0:3])

        batch_xs, batch_ys = X, Y

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 50 == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Setp: ", i, "Accuracy: ",sess.run(accuracy, feed_dict={x: X, y_: Y}))
            # print("Loss: ", sess.run(loss, feed_dict={x: X, y_: Y}))
            # print(sess.run(tf.argmax(y, 1), feed_dict={x: X}))

        if i % 1000 == 0:
            saver.save(sess, root+"softmax.ckpt")