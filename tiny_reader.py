import os
import numpy as np
import cv2
from numpy.random import randint
from keras.utils import np_utils
import h5py

num_train_data = 200 * 500
num_val_data = 10000
data_count_per_label = 500
img_shape = [64, 64, 3]


def import_hdf5(hdf5_path):
    h5f = h5py.File(hdf5_path, 'r')
    x_train = h5f['x_train'][:]
    y_train = h5f['y_train'][:]
    x_val = h5f['x_val'][:]
    y_val = h5f['y_val'][:]
    label_codes = h5f['label_codes_np'][:]
    h5f.close()
    return x_train, y_train, x_val, y_val, label_codes


class TinyImgReader:
    def __init__(self, root_path):
        self.tiny_root_path = root_path
        self.label_codes = self.__get_dir_list(os.path.join(root_path, 'train'))
        self.num_label = len(self.label_codes)
        self.all_img_name_list = []
        self.get_all_train_img_name(self.all_img_name_list)

    def get_label_code_list(self):
        return self.label_codes

    def get_num_class(self):
        return self.num_label

    def get_all_train_img_name(self, all_img_name_list):
        all_img_name_list.clear()
        for code in self.label_codes:
            path = os.path.join(self.tiny_root_path, "train", code, 'images')
            all_img_name_list.append(os.listdir(path))

    def __get_train_img_dir(self, label):
        return os.path.join(self.tiny_root_path, "train", self.label_codes[label], 'images')

    def load_all_val_img(self, scale_size=None):
        load_img_idx = 0

        if scale_size is None:
            x_val_shape = [num_val_data, 64, 64, 3]
        else:
            x_val_shape = [num_val_data, scale_size[0], scale_size[1], 3]

        code_to_label = dict(zip(self.label_codes, range(self.num_label)))
        val_annotations_path = os.path.join(self.tiny_root_path, 'val', 'val_annotations.txt')
        val_annotations_file = open(val_annotations_path, 'r')
        val_annotations_line = val_annotations_file.read()
        val_img_dir = os.path.join(self.tiny_root_path, 'val', 'images')
        x_val = np.zeros(x_val_shape, dtype=np.uint8)
        y_val = np.zeros(num_val_data, dtype=np.uint8)

        for line in val_annotations_line.splitlines():
            if load_img_idx % 500 == 0:
                print('Load validation image: ', load_img_idx, '/', num_val_data)
            pieces = line.strip().split()
            img_path = os.path.join(val_img_dir, pieces[0])
            img = cv2.imread(img_path)

            if scale_size is not None:
                x_val[load_img_idx] = cv2.resize(img, scale_size)

            y_val[load_img_idx] = code_to_label[pieces[1]]
            load_img_idx += 1

        print('Load validation data finished!')
        return x_val, y_val

    def load_all_train_img(self, scale_size=None):
        load_img_idx = 0

        if scale_size is None:
            x_train_shape = [num_train_data, 64, 64, 3]
        else:
            x_train_shape = [num_train_data, scale_size[0], scale_size[1], 3]

        x_train = np.zeros(x_train_shape, dtype=np.uint8)
        y_train = np.zeros(num_train_data, dtype=np.uint8)
        print(x_train.shape)

        for idx, img_name_list in enumerate(self.all_img_name_list):
            print('Load label: ', idx, self.label_codes[idx], '...')

            img_dir = os.path.join(self.tiny_root_path, "train", self.label_codes[idx], 'images')
            # TODO - Let y_train be vectorize implementation
            y_train[load_img_idx:load_img_idx + data_count_per_label - 1] = idx

            for img_name in img_name_list:
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)

                if scale_size is not None:
                    x_train[load_img_idx] = cv2.resize(img, scale_size)

                load_img_idx += 1

        print('Load training data finished!')
        return x_train, y_train

    # index have to smaller than data count.
    def load_img(self, label, img_index, is_normalize=True):
        img_dir = self.__get_train_img_dir(label)
        img_name = self.all_img_name_list[label][img_index]
        if is_normalize:
            return cv2.imread(os.path.join(img_dir, img_name)) / 255
        else:
            return cv2.imread(os.path.join(img_dir, img_name))

    def export_hdf5(self, scale_size=None):
        x_train, y_train = self.load_all_train_img(scale_size)
        x_val, y_val = self.load_all_val_img(scale_size)

        label_codes_np = np.chararray(self.num_label, 10)
        for i, name in enumerate(self.label_codes):
            label_codes_np[i] = self.label_codes[i]

        if scale_size is not None:
            h5f = h5py.File('training_data_' + str(scale_size[0]) + '_' + str(scale_size[1]) + '.h5', 'w')
        else:
            h5f = h5py.File('training_data.h5', 'w')

        h5f.create_dataset('x_train', data=x_train, compression="gzip", compression_opts=9)
        h5f.create_dataset('y_train', data=y_train, compression="gzip", compression_opts=9)
        h5f.create_dataset('x_val', data=x_val, compression="gzip", compression_opts=9)
        h5f.create_dataset('y_val', data=y_val, compression="gzip", compression_opts=9)
        h5f.create_dataset('label_codes_np', data=label_codes_np, compression="gzip", compression_opts=9)
        h5f.close()

        print('Export done')

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

