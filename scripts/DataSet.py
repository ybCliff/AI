import cv2
import numpy
import random

root = 'D:/graduation_project/workspace/dataset/'
trg = 'D:/graduation_project/workspace/dataset/UCF101_train_test_splits/train1_labels.txt'

split_dict = {'train1': 9537, 'train2': 9586, 'train3': 9624}
default_image_size = 224


class DataSet(object):
    def __init__(self,
                 dataset_name,
                 split_name,
                 frame_num,
                 label_type='onehot'):

        self._num_type = 101 if dataset_name is 'UCF' else 51
        self._num_frame = int(frame_num)
        self._num_video = int(split_dict[split_name])
        self._num_total_frame = self._num_frame * self._num_video
        self._index_frame = [int(i) for i in numpy.arange(self._num_total_frame)]
        self._epochs_completed = 0
        self._index_in_epoch_frame = 0
        self._label_type = label_type

        if dataset_name is 'UCF':
            self._path_to_read_labels = root + 'UCF101_train_test_splits/' + split_name + '_labels.txt'
            self._path_to_read_frames = root + 'UCF101_train_test_splits/' + split_name + '_spatial/'
        else:
            self._path_to_read_labels = root + 'HMDB51_train_test_splits/' + split_name + '_labels.txt'
            self._path_to_read_frames = root + 'HMDB51_train_test_splits/' + split_name + '_spatial/'

        file_to_read = open(self._path_to_read_labels, 'r')
        tmp = file_to_read.read()
        self._labels = [int(i) for i in tmp.split(' ')]
        file_to_read.close()

    def random_clip(self, img):
        width, height, _ = img.shape
        width_start = random.randint(0, width - default_image_size)
        height_start = random.randint(0, height - default_image_size)
        return img[width_start:width_start+default_image_size, height_start:height_start+default_image_size, :]

    def one_batch(self, index_frame):
        images = []
        labels = []

        for id in index_frame:
            video_id = int(id // self._num_frame)
            frame_id = int(id %  self._num_frame)
            label = self._labels[video_id]

            if self._label_type is 'onehot':
                tmp_labels = [0] * self._num_type
                tmp_labels[label - 1] = 1
                labels.append(tmp_labels)
            else:
                labels.append(label)

            frame_name = str(video_id) + '_' + str(frame_id) + '_' + str(label) + '.jpg'
            img = cv2.imread(self._path_to_read_frames + frame_name)

            if img is None:
                print(frame_name)
            else:
                images.append(self.random_clip(img))

        if len(images) == 0:
            return None, None
        images = numpy.array(images)
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        return images, numpy.array(labels)

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size > 0

        start = self._index_in_epoch_frame

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            numpy.random.shuffle(self._index_frame)
            #print(self._index_video)

        # Go to the next epoch
        if start + batch_size > self._num_total_frame:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_step_length = self._num_total_frame - start
            video_index_rest = self._index_frame[int(start):int(self._num_total_frame)]

            # Shuffle the data
            if shuffle:
                numpy.random.shuffle(self._index_frame)

            # Start next epoch
            start = 0
            self._index_in_epoch_frame = batch_size - rest_step_length
            end = self._index_in_epoch_frame
            return self.one_batch(video_index_rest + self._index_frame[int(start):int(end)])
        else:
            self._index_in_epoch_frame += batch_size
            end = self._index_in_epoch_frame
            return self.one_batch(self._index_frame[int(start):int(end)])

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def num_frames(self):
        return self._num_frame

    @property
    def index_frame(self):
        return self._index_frame

    @property
    def total_frame(self):
        return self._num_total_frame

    # def get_all_variables(self):
    #     print('            _num_type: ', self._num_type)
    #     print('           _num_frame: ', self._num_frame)
    #     print('           _num_video: ', self._num_video)
    #     print('    _epochs_completed: ', self._epochs_completed)
    #     print('_index_in_epoch_frame: ', self._index_in_epoch_frame)
    #     print('_index_in_epoch_video: ', self._index_in_epoch_video)


def splist(l, n):
    length = len(l)
    sz = length // n
    c = length % n
    lst = []
    i = 0
    while i < n:
        if i < c:
            bs = sz + 1
            lst.append(l[i*bs:i*bs+bs])
        else:
            lst.append(l[i*sz+c:i*sz+c+sz])
        i += 1
    return lst


class Test(object):
    def __init__(self,
                 dataset_name,
                 split_name,
                 frame_num):
        if dataset_name is 'UCF':
            self._path_to_read_video = root + 'UCF101_train_test_splits/' + split_name + '/'
        else:
            self._path_to_read_video = root + 'HMDB51_train_test_splits/' + split_name + '/'
        self._num_frame = frame_num


    def random_clip(self, img):
        width, height, _ = img.shape
        width_start = random.randint(0, width - default_image_size)
        height_start = random.randint(0, height - default_image_size)
        return img[width_start:width_start + default_image_size, height_start:height_start + default_image_size, :]

    def test_an_video(self, trg_root):
        images = []
        vc = cv2.VideoCapture(trg_root)  # 读入视频文件

        if not vc.isOpened():
            print('Open failure! exit')
            exit(0)

        total = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        sp = splist([i for i in range(int(total))], self._num_frame)
        # print('hahaha ', total)

        lst = []
        for i in range(self._num_frame):
            lst.append(random.sample(sp[i], 1))

        index = 0
        count = 0
        rval = True
        while rval and index < self._num_frame:  # 循环读取视频帧
            rval, frame = vc.read()
            if (count == lst[index][0]):
                if frame is not None:
                    images.append(self.random_clip(frame))
                index += 1
            count += 1
            cv2.waitKey(1)
        vc.release()

        images = numpy.array(images)
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        return images
