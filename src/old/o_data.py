import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CellDataset(Dataset):
    def __init__(self, filepath):
        super(CellDataset, self).__init__()

        """
        fh = open(txt, 'r')
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))
            # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        """
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def load_images(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name, -1)
        if img.dtype == 'uint16':
            img = img.astype(np.uint8)
        else:
            raise TypeError(
                'No such of img transfer type: {} for img'.format(img.dtype))
        images.append(img)
    images = np.array(images)
    return images


def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        return img.astype(np.uint8)
    else:
        raise TypeError(
            'No such of img transfer type: {} for img'.format(img.dtype))


def img_standardization(img):
    img = unit16b2uint8(img)
    """
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        img = np.tile(img, (1, 1, 3))
        return img
    elif len(img.shape) == 3:
        return img
    else:
        raise TypeError('The Depth of image large than 3 \n')
    """
    return img


def binaryzation(image):
    image[image > 0] = 1
    return image


SHOW_DATA = False


def get_dataset(width_in, height_in, width_out, height_out, binarize=True):

    # get train x
    absp = '/'.join(os.path.abspath(__file__).split('\\')[:-1]) + '/'
    train_x_path = absp + '../supplementary/dataset1/train/'
    train_x_list = [os.path.join(train_x_path, image)
                    for image in os.listdir(train_x_path)]
    train_X = load_images(train_x_list)

    # get train y
    train_y_path = absp + '../supplementary/dataset1/train_GT/SEG'
    train_y_list = [os.path.join(train_y_path, image)
                    for image in os.listdir(train_y_path)]
    train_Y = load_images(train_y_list)
    if binarize:
        train_Y = binaryzation(train_Y)  # for instance segmentation

    if SHOW_DATA:
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.imshow(train_x[3])
        plt.subplot(1, 2, 2)
        plt.imshow(train_y[3] * 255)
        plt.show()

    """
    result_path = absp + '../supplementary/dataset1/test_RES'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    """

    train_X, test_X, train_y, test_y = train_test_split(train_X, train_Y,
                                                        test_size=0.75,
                                                        random_state=0)

    return train_X, train_y, test_X, test_y


"""
class SaltDataset(Dataset):
    def __init__(self, image_list, mode, mask_list=None, fine_size=202, pad_left=0, pad_right=0):
        self.imagelist = image_list
        self.mode = mode
        self.masklist = mask_list
        self.fine_size = fine_size
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])

        if self.mode == 'train':
            mask = deepcopy(self.masklist[idx])
            label = np.where(mask.sum() == 0, 1.0, 0.0).astype(np.float32)

            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(
                    image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image, mask = do_center_pad2(
                    image, mask, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])

            return image, mask, label

        elif self.mode == 'val':
            mask = deepcopy(self.masklist[idx])

            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(
                    image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])

            return image, mask

        elif self.mode == 'test':
            if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(
                    self.fine_size, self.fine_size))

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])

            return image
"""
