# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.python.platform
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

input = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

tmp = tf.reshape(input, [-1, 28, 28, 1])
net = slim.conv2d(tmp, 32, [5, 5], scope='conv1')
# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# print(slim.get_model_variables())
# print(W)

net = slim.max_pool2d(net, [2, 2], scope='pool1')
net = slim.conv2d(net, 64, [5, 5], scope='conv2')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
net = slim.flatten(net)
net = slim.fully_connected(net, 1024, scope='fc')
net = slim.dropout(net, keep_prob, scope='dropout')
net = slim.fully_connected(net, 10, scope='output')

print(net.shape)
loss = slim.losses.softmax_cross_entropy(net, label)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(mnist.train.images.shape)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged_summary_op = tf.merge_all_summaries
    summary_writer = tf.train.SummaryWriter('D:/graduation_project/dummy_logs/mnist_logs', sess.graph)
    # 训练
    for i in range(1100):
        input_batch, label_batch = mnist.train.next_batch(batch_size=50)
        if i % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={input: input_batch, label: label_batch, keep_prob: 1.0})
            print ("step %d, training acc %g" % (i, train_accuracy))
            summary_str = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_str, i)
        train_step.run(feed_dict={input: input_batch, label: label_batch, keep_prob: 0.5})

    saver.save(sess, "D:/graduation_project/dummy_checkpoints/mnist.ckpt")


# # variables_to_restore = slim.get_model_variables()
# # variables_to_restore = {var for var in variables_to_restore}
# # restorer = tf.train.Saver(variables_to_restore)
# restorer = tf.train.Saver()
#
# # 如果一次性来做测试的话，可能占用的显存会比较多，所以测试的时候也可以设置较小的batch来看准确率
# test_acc_sum = tf.Variable(0.0)
# batch_acc = tf.placeholder(tf.float32)
# new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
# update = tf.assign(test_acc_sum, new_test_acc_sum)
#
# with tf.Session() as sess:
#     #sess.run(tf.initialize_variables(test_acc_sum))
#     restorer.restore(sess, "D:/graduation_project/dummy_checkpoints/mnist.ckpt")
#     # print(sess.run(slim.get_model_variables()))
#     sess.run(tf.initialize_variables([test_acc_sum]))
#     # 全部训练完了再做测试，batch_size=100
#     for i in range(100):
#         input_batch, label_batch = mnist.test.next_batch(batch_size=100)
#         test_acc = accuracy.eval(feed_dict={input: input_batch, label: label_batch, keep_prob: 1.0})
#         update.eval(feed_dict={batch_acc: test_acc})
#         if (i+1) % 20 == 0:
#             print ("testing step %d, test_acc_sum %g" % (i+1, test_acc_sum.eval()))
#     print (" test_accuracy %g" % (test_acc_sum.eval() / 100.0))

