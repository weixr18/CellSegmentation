import os
import gc

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from unet import UNet
from data import get_dataset


class Trainer():
    def __init__(self):
        pass

    def setup(self, data_dir="", mask_dir="", valid_rate=0.1,
              criterion=None, optimizer=None, hyper_params=None
              ):

        self.train_dataset, self.valid_dataset = get_dataset(
            data_dir, mask_dir, valid_rate)

        self.unet = UNet(n_channels=1, n_classes=2,)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.unet.parameters(), lr=self.hyper_params["learning_rate"], momentum=0.99)

        self.hyper_params = hyper_params

        self.train_data_loader = DataLoader(
            dataset=self.train_dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=True
        )
        self.valid_data_loader = DataLoader(
            dataset=self.valid_dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )

    def run(self):

        # TODO: rewrite the train flow with the data_loders

        epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)

        for _ in range(epochs):
            total_loss = 0
            for i in tqdm(range(epoch_iter), ascii=True, ncols=80):
                batch_train_x = torch.from_numpy(
                    x_train[i * batch_size: (i + 1) * batch_size]).float()
                batch_train_y = torch.from_numpy(
                    y_train[i*batch_size:(i + 1)*batch_size]).long()

                if (len(batch_train_x.size()) == 3):
                    batch_train_x = batch_train_x.unsqueeze(1)
                if (len(batch_train_y.size()) == 3):
                    batch_train_y = batch_train_y.unsqueeze(1)

                if use_gpu:
                    batch_train_x = batch_train_x.cuda()
                    batch_train_y = batch_train_y.cuda()
                batch_loss = self.train_step(
                    batch_train_x, batch_train_y, optimizer, criterion, unet, width_out, height_out)
                total_loss += batch_loss
            if (_+1) % epoch_lapse == 0:
                val_loss = get_val_loss(x_val, y_val, width_out, height_out,
                                        unet, batch_size, use_gpu)
                print("Total loss in epoch %f : %f and validation loss : %f" %
                      (_+1, total_loss, val_loss))
        gc.collect()
        pass

    def train_step(self, inputs, labels, optimizer, criterion, unet, width_out, height_out):
        optimizer.zero_grad()

        outputs = unet(inputs)
        outputs = outputs.permute(0, 2, 3, 1)

        m = outputs.shape[0]
        outputs = outputs.reshape(m*width_out*height_out, 2)
        labels = labels.reshape(m * width_out * height_out)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        return loss

    def save_module(self, save_path):
        # TODO: save the module in the save_path
        pass
