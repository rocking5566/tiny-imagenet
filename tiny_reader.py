import os
import numpy as np
import cv2
from numpy.random import randint
from keras.utils import np_utils


data_count_per_label = 500
x_train_shape = [200 * 500, 64, 64, 3]
img_shape = [64, 64, 3]


class TinyTrainReader:
    def __init__(self, train_dir):
        self.tiny_root_path = train_dir
        self.label_codes = self.__get_dir_list(train_dir)
        self.num_label = len(self.label_codes)
        self.all_img_name_list = []
        self.get_all_img_name(self.all_img_name_list)

    def get_train_img_label_code_list(self):
        return self.label_codes

    def get_num_class(self):
        return self.num_label

    def get_all_img_name(self, all_img_name_list):
        all_img_name_list.clear()
        for code in self.label_codes:
            path = os.path.join(self.tiny_root_path, code, 'images')
            all_img_name_list.append(os.listdir(path))

    def get_img_dir(self, label):
        return os.path.join(self.tiny_root_path, self.label_codes[label], 'images')

    def load_all_img(self):
        load_img_idx = 0
        x_train = np.zeros(x_train_shape)
        y_train = np.zeros(self.num_label * data_count_per_label)

        for idx, img_name_list in enumerate(self.all_img_name_list):
            print('Load label: ', idx, self.label_codes[idx], '...')

            img_dir = os.path.join(self.tiny_root_path, self.label_codes[idx], 'images')
            # TODO - Let y_train be vectorize implementation
            y_train[load_img_idx:load_img_idx + data_count_per_label - 1] = idx

            for img_name in img_name_list:
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                x_train[load_img_idx] = img
                load_img_idx += 1
        return x_train, y_train

    # index have to smaller than data count.
    def load_img(self, label, img_index, is_normalize = True):
        img_dir = self.get_img_dir(label)
        img_name = self.all_img_name_list[label][img_index]
        if is_normalize:
            return cv2.imread(os.path.join(img_dir, img_name)) / 255
        else:
            return cv2.imread(os.path.join(img_dir, img_name))

    def generate_data(self, batch_size):
        while 1:
            x = np.zeros([batch_size, img_shape[0], img_shape[1], img_shape[2]])
            y = np.zeros([batch_size])

            index = randint(self.num_label * data_count_per_label, size=batch_size)
            label = (index / data_count_per_label).astype(int)
            img_index = index % data_count_per_label

            for i in range(batch_size):
                x[i] = self.load_img(label[i], img_index[i])
                y[i] = label[i]

            y = np_utils.to_categorical(y, num_classes=self.num_label)    #one hot
            yield (x,  y)

    @staticmethod
    def __get_dir_list(path):
        for (_, dir_names, _) in os.walk(path):
            return dir_names
