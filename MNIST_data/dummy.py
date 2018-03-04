import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
input = tf.Variable(tf.random_normal([1,2,2,1]))
filter = tf.Variable(tf.random_normal([2,2,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(input.eval())
    # print(input.shape)
    a = input[0][0][0][0].eval()
    b = input[0][0][1][0].eval()
    c = input[0][1][0][0].eval()
    d = input[0][1][1][0].eval()
    print(a, b, c, d)


    print('--------------')
    print(filter.eval())
    # print(filter.shape)
    e = filter[0][0][0][0].eval()
    f = filter[0][1][0][0].eval()
    g = filter[1][0][0][0].eval()
    h = filter[1][1][0][0].eval()
    print(e, f, g, h)

    print('--------------')
    print(op2.eval())
    print('--------------')
    print(op.eval())
    print('--------------')
    arr1 = np.array([[a, b], [c, d]])
    arr2 = np.array([[e, f], [g, h]])
    print(sum(sum(arr1 * arr2)))
    print(e * b + g * d)
    print(sum(arr1[1] * arr2[0]))
    print(e * d)

