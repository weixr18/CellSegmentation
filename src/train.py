# train
import os
import gc

import cv2
import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


from unet import UNet
from data import get_dataset
from validate import Validator

SHOW_NET = False


class Trainer():
    def __init__(self):
        pass

    def setup(self, valid_rate=0.1, use_cuda=True,
              cell_dir="", mask_dir="", module_save_dir="", tmp_dir="",
              criterion=None, optimizer=None, hyper_params=None,
              ):
        """setup the module"""
        self.train_dataset, self.valid_dataset = get_dataset(
            cell_dir, mask_dir, valid_rate, tmp_dir)

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

        self.use_cuda = use_cuda
        self.unet = UNet(n_channels=1, n_classes=2,)
        if use_cuda:
            self.unet = self.unet.cuda()
        if SHOW_NET:
            from torchsummary import summary
            batch_size = self.hyper_params["batch_size"]
            summary(self.unet, (batch_size, 628, 628))

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.unet.parameters(), lr=self.hyper_params["learning_rate"], momentum=0.99)
        self.module_save_dir = module_save_dir

        self.v = Validator(unet=self.unet,
                           hyper_params=hyper_params,
                           use_cuda=use_cuda,
                           data_loader=self.valid_data_loader)

    def train(self):
        """train the model"""
        epochs = self.hyper_params["epochs"]
        epoch_lapse = self.hyper_params["epoch_lapse"]
        batch_size = self.hyper_params["batch_size"]
        epoch_save = self.hyper_params["epoch_save"]
        width_out = 628
        height_out = 628

        for _ in range(epochs):
            total_loss = 0
            for data in tqdm(self.train_data_loader, ascii=True, ncols=120):

                batch_train_x, batch_train_y = data
                batch_train_y = batch_train_y.long()
                batch_train_y[batch_train_y > 0] = 1  # important!!!
                if (len(batch_train_x.size()) == 3):
                    batch_train_x = batch_train_x.unsqueeze(1)
                if (len(batch_train_y.size()) == 3):
                    batch_train_y = batch_train_y.unsqueeze(1)

                if self.use_cuda:
                    batch_train_x = batch_train_x.cuda()
                    batch_train_y = batch_train_y.cuda()

                batch_loss = self.train_step(
                    batch_train_x, batch_train_y,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    unet=self.unet,
                    width_out=width_out,
                    height_out=height_out,
                    batch_size=batch_size)

                total_loss += batch_loss

            if (_+1) % epoch_lapse == 0:
                val_acc = self.v.validate()
                print("Total loss in epoch %d : %f and validation accuracy : %f" %
                      (_ + 1, total_loss, val_acc))

            if (_+1) % epoch_save == 0:
                self.save_module(name_else="epoch-" + str(_ + 1))
                print("MODULE SAVED.")
        gc.collect()
        pass

    def train_step(self, inputs, labels, optimizer,
                   criterion, unet, batch_size,
                   width_out, height_out):
        optimizer.zero_grad()
        outputs = unet(inputs)
        outputs = outputs.permute(0, 2, 3, 1)

        outputs = outputs.reshape(batch_size * width_out * height_out, 2)
        labels = labels.reshape(batch_size * width_out * height_out)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        return loss

    def save_module(self, name_else=""):
        import datetime
        module_save_dir = self.module_save_dir
        filename = 'unet-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + \
            name_else + '.pth'
        torch.save(self.unet.state_dict(), module_save_dir + filename)
        pass
