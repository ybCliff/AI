import sys
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.training import saver as tf_saver
from scripts import DataSet
import urllib.request
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("D:/graduation_project/workspace/models/research/slim")
from scripts import my_vgg_16 as vgg

#######################################################
tf.app.flags.DEFINE_float(
    'learning_rate', 1e-4, 'Train learing rate')
tf.app.flags.DEFINE_float(
    'momentum', 0.9, '')
tf.app.flags.DEFINE_integer(
    'number_of_steps', 50, 'Max steps while training a batch')
#######################################################

checkpoint_path = 'D:/graduation_project/checkpoints/vgg_16.ckpt'
save_model_path = 'D:/graduation_project/workspace/checkpoints'
new_model_checkpoint_path = 'D:/graduation_project/workspace/checkpoints/checkpoint'

FLAGS = tf.app.flags.FLAGS
image_size = vgg.vgg_16.default_image_size
# input = tf.placeholder(tf.int32, [-1, image_size, image_size, 3])
# label = tf.placeholder(tf.float32, [-1, 101])

###############################################################################
def init_fn_part(): #从checkpoint读入网络权值
    variables_to_restore = slim.get_variables_to_restore(exclude=["vgg_16/fc8"])
    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

def init_fc8(): #初始化最后一层
    return tf.variables_initializer(slim.get_variables("vgg_16/fc8"))

def init_fn_full():
    return slim.assign_from_checkpoint_fn(new_model_checkpoint_path, slim.get_variables("vgg_16"))

# def init_fn_full():
#     variables_to_restore1 = slim.get_variables_to_restore(exclude=["vgg_16/fc8"])
#     variables_to_restore2 = slim.get_variables_to_restore(slim.get_variables("vgg_16/fc8"))
#     saver1 = tf_saver.Saver(variables_to_restore1)
#     saver2 = tf_saver.Saver(variables_to_restore2)
#     def callback(session):
#         saver1.restore(session, checkpoint_path)
#         saver2.restore(session, )
#
#     return callback
#     return [slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore1),slim.assign_from_checkpoint_fn(new_model_checkpoint_path, variables_to_restore2)]
###############################################################################


def train_a_batch(input, label, load = True, new = False):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(input,
                               num_classes=101,
                               is_training=True)

    probabilities = tf.nn.softmax(logits)
    label = tf.convert_to_tensor(label)

    # Define the loss:
    loss = slim.losses.softmax_cross_entropy(logits, label)
    # slim.losses.log_loss(tf.nn.softmax(logits), label)
    # total_loss = slim.losses.get_total_loss()

    # Define the optimizer:
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)

    # Create the train_op
    variables_to_train = slim.get_variables('vgg_16/fc8')
    train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=variables_to_train)
    # train_op = slim.learning.create_train_op(loss, optimizer)

    # Define saver
    # saver = tf.train.Saver(var_list=slim.get_variables('vgg_16/fc8'), max_to_keep=3)
    saver = tf.train.Saver(max_to_keep=3)
    # Run training.
    if load is True:
        if new is True:
            slim.learning.train(train_op,
                                init_op=init_fc8(),
                                local_init_op=None,
                                init_fn=init_fn_part(),
                                logdir=save_model_path,
                                number_of_steps=FLAGS.number_of_steps,
                                summary_writer=None,
                                saver=saver)
        else:
            slim.learning.train(train_op,
                                init_op=None,
                                local_init_op=None,
                                init_fn=init_fn_full(),
                                logdir=save_model_path,
                                number_of_steps=FLAGS.number_of_steps,
                                summary_writer=None,
                                saver=saver)
    else:
        slim.learning.train(train_op,
                            init_op=None,
                            local_init_op=None,
                            init_fn=None,
                            logdir=save_model_path,
                            number_of_steps=FLAGS.number_of_steps,
                            summary_writer=None,
                            saver=saver)


if __name__ == '__main__':
    train = DataSet.DataSet('UCF', 'train1', 25)
    total_time = 0
    for i in range(12000):
        test_time = time.time()
        tf.reset_default_graph()
        print('Reset：%.2fs'%(time.time()-test_time))
        begin_time = time.time()
        input, label = train.next_batch(24)
        if i == 0:
            train_a_batch(input, label, load=True, new=False)
        else:
            train_a_batch(input, label, load=False, new=False)
        end_time = time.time()
        total_time += (end_time - begin_time)
        print('index: %.0f   Total: %.2f   loop: %.2f' % (i, total_time, end_time-begin_time))