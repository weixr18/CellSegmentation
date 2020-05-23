import os
import gc

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def load_file(filename):
    # TODO: 根据filename读出PIL格式的图像
    pass


class CellDataset(Dataset):
    def __init__(self, txtpath, transform, dataloader):
        super(CellDataset, self).__init__()

        data_paths = []
        with open(txtpath, 'r') as fh:
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = line.split()    # 0和1分别是cell和mask路径
                data_paths.append(words[0], words[1])

        self.data_paths = data_paths
        self.transform = transform
        self.dataloader = dataloader
        pass

    def __getitem__(self, index):

        cell_path, mask_path = self.data_paths[index]
        cell = self.loader(cell_path)
        mask = self.loader(mask_path)
        if self.transform is not None:
            cell, mask = self.transform(cell, mask)
        return cell, mask

    def __len__(self):
        return len(self.data_paths)


def get_dataset(data_dir, mask_dir, valid_rate):

    # TODO:
    # 1. 生成data_dir的文件列表data_file_list
    # 2. 根据valid_rate把data_file_list分为两部分
    # 3. 将train_list存成"train_data.txt"
    # 4. 将valid_list存成"valid_data.txt"
    # 5. 分别获取两个dataset

    train_data_txt = ""
    valid_data_txt = ""

    train_dataset = CellDataset(train_data_txt)
    valid_dataset = CellDataset(valid_data_txt)

    return train_dataset, valid_dataset
