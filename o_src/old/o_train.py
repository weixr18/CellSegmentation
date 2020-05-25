import os
import gc

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet import UNet
from data import get_dataset
from validate import get_val_loss


def train_step(inputs, labels, optimizer, criterion, unet, width_out, height_out):
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


def train(unet, batch_size, epochs, epoch_lapse,
          threshold, learning_rate, criterion,
          optimizer, x_train, y_train, x_val, y_val,
          width_out, height_out):
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
            batch_loss = train_step(
                batch_train_x, batch_train_y, optimizer, criterion, unet, width_out, height_out)
            total_loss += batch_loss
        if (_+1) % epoch_lapse == 0:
            val_loss = get_val_loss(x_val, y_val, width_out, height_out,
                                    unet, batch_size, use_gpu)
            print("Total loss in epoch %f : %f and validation loss : %f" %
                  (_+1, total_loss, val_loss))
    gc.collect()


TRAIN = False

if __name__ == "__main__":

    width_in = 628
    height_in = 628
    width_out = 628
    height_out = 628
    batch_size = 1
    epochs = 1
    epoch_lapse = 50
    threshold = 0.5
    learning_rate = 0.01
    use_gpu = True

    x_train, y_train, x_val, y_val = get_dataset(
        width_in, height_in, width_out, height_out, binarize=TRAIN)

    if TRAIN:
        unet = UNet(n_channels=1, n_classes=2,)
        if use_gpu:
            unet = unet.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.99)
        train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate,
              criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out)
    else:
        modelPath = r"D:\Machine_Learning\Codes\CellSegment\save\unet-20200520185855.pth"

        unet = UNet(n_channels=1, n_classes=2,)
        unet.load_state_dict(torch.load(modelPath))
        if use_gpu:
            unet = unet.cuda()

        val_loss = get_val_loss(x_val, y_val, width_out, height_out,
                                unet, batch_size, use_gpu)

        pass
