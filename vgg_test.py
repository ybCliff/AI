import sys
import time
import tensorflow as tf
import urllib.request
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("D:/graduation_project/workspace/models/research/slim")
from scripts import my_vgg_16 as vgg
from datasets import imagenet
from preprocessing import vgg_preprocessing

checkpoints_dir = 'D:/graduation_project/checkpoints/vgg_16.ckpt'
slim = tf.contrib.slim

# 网络模型的输入图像有默认的尺寸
# 因此，我们需要先调整输入图片的尺寸
image_size = vgg.vgg_16.default_image_size
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

    # 可以批量导入图像，第一个维度指定每批图片的张数，我们每次只导入一张图片
    print(processed_image.shape)
    processed_images  = tf.expand_dims(processed_image, 0)
    print(processed_images.shape)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=True)

    print('Trainable variables: ', tf.trainable_variables())
    # 我们在输出层使用softmax函数，使输出项是概率值
    probabilities = tf.nn.softmax(logits)

    # 创建一个函数，从checkpoint读入网络权值
    init_fn = vgg.get_init_fn(checkpoint_path=checkpoints_dir, checkpoint_exclude_scopes='vgg_16/fc8')

    print(slim.get_model_variables())
    variable_to_train = vgg.get_variables_to_train('vgg_16/fc8')
    print(variable_to_train)
    # with tf.Session() as sess:
    #     #初始化最后一层
    #     begin_time = time.time()
    #     sess.run(tf.variables_initializer(slim.get_model_variables('vgg_16/fc8')))
    #     end_time = time.time()
    #     print('Initial fc8 time: ', end_time-begin_time, 'ms')
    #
    #     # 加载权值
    #     init_fn(sess)
    #     # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
    #     np_image, network_input, probabilities = sess.run([image,
    #                                                        processed_image,
    #                                                        probabilities])
    #     probabilities = probabilities[0, 0:]
    #     sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
    #                                         key=lambda x:x[1])]
    #
    # names = imagenet.create_readable_names_for_imagenet_labels()
    # for i in range(5):
    #     index = sorted_inds[i]
    #     # 打印top5的预测类别和相应的概率值。
    #     print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))
    #



