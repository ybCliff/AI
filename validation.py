from scripts import my_vgg_16 as vgg
from scripts import DataSet
import tensorflow as tf
import random
import cv2
import time
import os, sys, shutil
import random
from collections import defaultdict

def get_counts(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts


def top_counts(count_dict, n):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

slim = tf.contrib.slim
source_folder = 'D:/graduation_project/workspace/dataset/UCF101_train_test_splits/test1/'
checkpoints_dir = "D:/graduation_project/workspace/checkpoints\\model.ckpt-10227"
file_list = os.listdir(source_folder)

input = tf.placeholder(tf.float32, [None, 224, 224, 3])

with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(input,
                           num_classes=101,
                           is_training=False)

probabilities = tf.nn.softmax(logits)
print(slim.get_model_variables())
# 从checkpoint读入网络权值
init_fn = slim.assign_from_checkpoint_fn(checkpoints_dir, slim.get_model_variables('vgg_16'))

show_detail = False
right_count = 0

with tf.Session() as sess:
    # 加载权值
    begin_time = time.time()
    init_fn(sess)
    print('Loading: %.2fs' % (time.time()-begin_time))

    test = DataSet.Test('UCF101', 'test1', 10)
    begin_time = time.time()
    for k in range(len(file_list)):
        file = file_list[k]
        ground_true = (file.split('.')[0]).split('_')[1]
        input_batch = test.test_an_video(source_folder + file)
        result = probabilities.eval(feed_dict={input: input_batch})

        if show_detail:
            print('=====================  ' + str(k) + '  ==================')
        seq = []
        for x in range(result.shape[0]):
            prob = result[x, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-prob),
                                                key=lambda x: x[1])]
            if show_detail:
                print('Frame %2s: '% x, end="")
            for i in range(5):
                index = sorted_inds[i]
                # 打印top5的预测类别和相应的概率值。
                if show_detail:
                    print('[%3s:%0.2f]  ' % (index+1,prob[index]), end="")
                if i == 0:
                    seq.append(index+1)
            if show_detail:
                print('')

        if show_detail:
            print('Ground true => %s' % ground_true)
        tmp = top_counts(get_counts(seq), 1)
        if int(tmp[0][1]) == int(ground_true):
            right_count += 1
        if show_detail:
            print(tmp)
        if k % 100 == 0:
            print('====> %5.0f: %.2fs'%(k, time.time()-begin_time))

    print(right_count)
    print(len(file_list))
    print('Accuracy: .2f'%(right_count / len(file_list)))