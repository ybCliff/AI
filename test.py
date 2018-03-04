import sys
import numpy as np
import os
import tensorflow as tf
from skimage import io
import urllib.request
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("D:/graduation_project/workspace/models/research/slim")

from matplotlib import pyplot as plt
from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing
import time
checkpoints_dir = 'D:/graduation_project/checkpoints'

slim = tf.contrib.slim

# 网络模型的输入图像有默认的尺寸
# 因此，我们需要先调整输入图片的尺寸
image_size = vgg.vgg_16.default_image_size
sess2 = tf.Session()
with tf.Graph().as_default():
    # 读取图片
    # url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/First_Student_IC_school_bus_202076.jpg")
    url = ('https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1512416144146&di=69f1c6faea06cd374b3d0808ab9acde8&imgtype=0&src=http%3A%2F%2Fphotos.pkone.cn%2Fwx%2FSHIRE_IMAGESIGN_MEDIUM%2F2009%2F04%2F18%2F216ab01a-a23a-4ea2-8677-1b889fa063b1%2F24d29263-8aaf-45a4-96d0-72bd9747bfd2.jpg')
    # Open specified url and load image as a string
    image_string = urllib.request.urlopen(url).read()

    # image_string = io.imread('D:/graduation_project/workspace/image_data/First_Student_IC_school_bus_202076.jpg').tostring()

    # 将图片解码成jpeg格式
    image = tf.image.decode_jpeg(image_string, channels=3)
    # print(type(image))
    # print(image)
    # 对图片做缩放操作，保持长宽比例不变，裁剪得到图片中央的区域
    # 裁剪后的图片大小等于网络模型的默认尺寸
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)

    # 可以批量导入图像
    # 第一个维度指定每批图片的张数
    # 我们每次只导入一张图片
    # print(type(processed_image))
    # plt.figure()
    # plt.imshow(processed_image.astype(np.uint8))
    # plt.suptitle("ori image", fontsize=14, fontweight='bold')
    # plt.axis('off')
    # plt.show()
    print(processed_image.shape)
    print(type(processed_image))
    processed_images  = tf.expand_dims(processed_image, 0)
    print(processed_images.shape)
    print(type(processed_images))
    print(processed_images[0])
    # plt.figure()
    # plt.imshow(processed_image.astype(np.uint8))
    # plt.suptitle("processed image", fontsize=14, fontweight='bold')
    # plt.axis('off')
    # plt.show()
    # 创建模型，使用默认的arg scope参数
    # arg_scope是slim library的一个常用参数
    # 可以设置它指定网络层的参数，比如stride, padding 等等。
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)

    # 我们在输出层使用softmax函数，使输出项是概率值
    probabilities = tf.nn.softmax(logits)

    # 创建一个函数，从checkpoint读入网络权值
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    print(slim.get_model_variables())
    time1 = time.time()
    # saver = tf.train.Saver()
    saver = tf.train.Saver(var_list=slim.get_variables('vgg_16/fc8'))
    with tf.Session() as sess:

        # 加载权值
        time2 = time.time()
        init_fn(sess)
        saver.save(sess, "D:/graduation_project/dummy_checkpoints/save_test.ckpt")
        print('Time1: ', time.time() - time1)
        print('Time2：', time.time() - time2)
    #     # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
    #     np_image, network_input, probabilities = sess.run([image,
    #                                                        processed_image,
    #                                                        probabilities])
    #     print(probabilities.shape, type(probabilities))
    #     probabilities = probabilities[0, 0:]
    #     print(probabilities.shape, type(probabilities))
    #     sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
    #                                         key=lambda x:x[1])]
    # # 显示下载的图片
    # # plt.figure()
    # # plt.imshow(np_image)
    # # print(type(np_image.astype(np.uint8)))
    # # plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    # # plt.axis('off')
    # # plt.show()
    # #
    # # # 显示最终传入网络模型的图片
    # # # 图像的像素值做了[-1, 1]的归一化
    # # # to show the image.
    # # plt.imshow(network_input / (network_input.max() - network_input.min()))
    # # plt.suptitle("Resized, Cropped and Mean-Centered in put to network",
    # #              fontsize=14, fontweight='bold')
    # # plt.axis('off')
    # # plt.show()
    #
    # names = imagenet.create_readable_names_for_imagenet_labels()
    # for i in range(5):
    #     index = sorted_inds[i]
    #     # 打印top5的预测类别和相应的概率值。
    #     print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))
    #
    # res = slim.get_model_variables()


