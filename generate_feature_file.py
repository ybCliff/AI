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
txtName = 'D:/graduation_project/workspace/dataset/UCF101_train_test_splits/train1_feature_files/vgg16_fc7/'

train = DataSet.DataSet('UCF', 'train3', 25, label_type='notonehot')
total_frame = train.total_frame
myinput = tf.placeholder(tf.float32, [None, 224, 224, 3])
logits, _ = vgg.vgg_16(myinput, num_classes=1000, is_training=False, jud='fc7')
init = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore(include=["vgg_16"]))

beginTime = time.time()
pre = 0
post = 9999
status = True
with tf.Session() as sess:
    init(sess)
    for k in range(total_frame):
        if status is True:
            nowTxtName = txtName + str(pre) + '_' + str(post) + '.txt'
            f = open(nowTxtName, 'w')
            status = False
        inputa, label = train.next_batch(1)
        if(inputa is not None):
            logitsResult = logits.eval(feed_dict={myinput: inputa})
            batch = logitsResult.shape[0]
            entry = logitsResult.shape[3]
            logitsResult = numpy.reshape(logitsResult,[batch, entry])

            tmp = ','.join([str(i) for i in logitsResult[0]])
            tmp += (' '+str(label[0])+'\n')
            f.write(tmp)
        if k % 100 == 0:
            print('%.0lf ===> %.2lfs' %(k, time.time() - beginTime))
        if k == post:
            f.close()
            status = True
            pre += 10000
            post = min(post+10000, total_frame)