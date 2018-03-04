import sys
import time
import tensorflow as tf
from scripts import DataSet
import numpy
import urllib.request
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("D:/graduation_project/workspace/models/research/slim")
from scripts import my_vgg_16 as vgg
slim = tf.contrib.slim

checkpoint_path = 'D:/graduation_project/checkpoints/vgg_16.ckpt'
new_model_checkpoint_path = 'D:/graduation_project/workspace/checkpoints'

train = DataSet.DataSet('UCF', 'train1', 25)
input, label = train.next_batch(2)

logits, _ = vgg.vgg_16(input, num_classes=1000, is_training=False, jud='fc7')
init1 = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore(include=["vgg_16"]))
# init1 = slim.assign_from_checkpoint_fn(new_model_checkpoint_path + '\\model.ckpt-107', slim.get_variables("vgg_16"))
# init2 = slim.assign_from_checkpoint_fn(new_model_checkpoint_path + '\\model.ckpt-108', slim.get_variables("vgg_16"))

with tf.Session() as sess:
    # init1(sess)
    # fc8_biases = slim.get_variables("vgg_16/fc8/biases")
    # fc7_biases = slim.get_variables("vgg_16/fc7/biases")
    # print(fc8_biases)
    # print('fc8 biases pre: ', sess.run(fc8_biases[0:10]))
    # print('fc7 biases pre: ', sess.run(fc7_biases[0:10]))
    #
    # init2(sess)
    # fc8_biases_now = slim.get_variables("vgg_16/fc8/biases")
    # fc7_biases_now = slim.get_variables("vgg_16/fc7/biases")
    # print('fc8 biases pre: ', sess.run(fc8_biases_now[0:10]))
    # print('fc7 biases pre: ', sess.run(fc7_biases_now[0:10]))

    init1(sess)
    fc7_biases = slim.get_variables("vgg_16/fc7/biases")
    print(fc7_biases)
    tmp = sess.run(fc7_biases[0])
    print(type(tmp))
    print('fc7 biases: ', tmp)
    print(tmp.shape)


    logitsResult = logits.eval()
    print(type(logitsResult))
    batch = logitsResult.shape[0]
    entry = logitsResult.shape[3]
    logitsResult = numpy.reshape(logitsResult,[batch, entry])
    print(logitsResult.shape)
    print(logitsResult[0, :])
