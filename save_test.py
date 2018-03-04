import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import numpy as np
#保存时dtype类型要一致，一般使用float32，另外要定义变量名
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "D:/graduation_project/dummy_checkpoints/save_test.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))