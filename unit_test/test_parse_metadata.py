from tiny_reader import TinyTrainReader
import os

tiny_image_net_200_dir = '/src/dataset/imagenet/tiny-imagenet-200'
train_dir = os.path.join(tiny_image_net_200_dir, 'train')
val_ans = os.path.join(tiny_image_net_200_dir, 'val/val_annotations.txt')
reader = TinyTrainReader(train_dir)


def test_get_train_img_label_code_list():
    code_list = reader.get_train_img_label_code_list()
    assert(code_list.count('n01443537') == 1)
    assert(code_list.count('n02892201') == 1)
    assert(code_list.count('n04149813') == 1)
    assert(code_list.count('xyz') == 0)


def test_get_num_class():
    assert(reader.get_num_class() == 200)


def test_data_path():
    for i in range(reader.get_num_class()):
        assert(len(reader.all_img_name_list[i]) == 500)


def test_get_img_dir():
    assert(reader.get_img_dir(0) == '/src/dataset/imagenet/tiny-imagenet-200/train/n01443537/images')


def test_generate_data():
    for idx, data in enumerate(reader.generate_data(3)):
        if idx > 2:
            break

        assert(len(data) == 2)
        assert(data[0].shape == (3, 64, 64, 3))
        assert(data[1].shape == (3, reader.num_label))
