# data
import os
import gc

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

"""
flip
rotation
zoom
shift
fill_mode=reflect
"""

data_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
])


def load_file(filename):
    img = cv2.imread(filename, -1)
    img = img.astype(np.float32)
    return img


class TestSet(Dataset):
    def __init__(self, cell_dir, data_reader=load_file):

        super(TestSet, self).__init__()
        data_paths = os.listdir(cell_dir)
        self.data_paths = []
        for p in data_paths:
            self.data_paths.append(cell_dir + p)

        self.data_reader = data_reader
        pass

    def __getitem__(self, index):
        cell_path = self.data_paths[index]
        cell = self.data_reader(cell_path)

        # Normalization
        cell = cell - cell.min()
        cell = cell / cell.max() * 255
        return cell

    def __len__(self):
        return len(self.data_paths)

    pass


class CellDataset(Dataset):
    def __init__(self, txtpath, transform=data_transforms, data_reader=None):
        super(CellDataset, self).__init__()

        data_paths = []
        with open(txtpath, 'r') as fh:
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = line.split()    # 0和1分别是cell和mask路径
                data_paths.append((words[0], words[1]))

        self.data_paths = data_paths
        self.transform = transform
        self.data_reader = data_reader
        pass

    def __getitem__(self, index):

        cell_path, mask_path = self.data_paths[index]
        cell = self.data_reader(cell_path)
        mask = self.data_reader(mask_path)

        # Normalization
        cell = cell - cell.min()
        cell = cell / cell.max() * 255

        if self.transform is not None:
            img = np.uint8([cell, mask, mask]).transpose(1, 2, 0)
            img = Image.fromarray(img)
            img = self.transform(img)
            cell = img[0]
            mask = img[1] * 255

        return cell, mask

    def __len__(self):
        return len(self.data_paths)


def get_dataset(cell_dir, mask_dir, valid_rate, tmp_dir, use_exist=True):

    valid_txt = tmp_dir + "valid_data.txt"
    train_txt = tmp_dir + "train_data.txt"

    use_exist = use_exist and os.path.isfile(
        valid_txt) and os.path.isfile(train_txt)

    if not use_exist:
        # generate list of file names
        cell_list = [os.path.join(cell_dir, image)
                     for image in os.listdir(cell_dir)]
        mask_list = [os.path.join(mask_dir, image)
                     for image in os.listdir(mask_dir)]

        # separate the lists according to valid_rate
        sample_size = len(cell_list)
        valid_size = int(sample_size * valid_rate)
        valid_index = np.random.choice(
            a=sample_size, size=valid_size, replace=False, p=None)

        # save the lists in txt files
        with open(valid_txt, "a+") as f:
            for i in valid_index:
                f.write(cell_list[i] + " " + mask_list[i] + '\n')

        with open(train_txt, "a+") as f:
            for i in range(sample_size):
                if i not in valid_index:
                    f.write(cell_list[i] + " " + mask_list[i] + '\n')

    # get the Dataset objects
    train_dataset = CellDataset(train_txt, data_reader=load_file)
    valid_dataset = CellDataset(valid_txt, data_reader=load_file)

    return train_dataset, valid_dataset
