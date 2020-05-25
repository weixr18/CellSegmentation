import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy


def get_val_loss(x_val, y_val, width_out, height_out,
                 unet, batch_size=1, use_gpu=True):

    epoch_iter = np.ceil(x_val.shape[0] / batch_size).astype(int)

    j_scores = []

    predict_ys = []
    for i in range(epoch_iter):
        # preprocess
        batch_val_x = torch.from_numpy(
            x_val[i*batch_size:(i + 1)*batch_size]).float()
        if (len(batch_val_x.size()) == 3):
            batch_val_x = batch_val_x.unsqueeze(1)
        if use_gpu:
            batch_val_x = batch_val_x.cuda()

        # get predict
        batch_predict_y = unet(batch_val_x)
        batch_predict_y = batch_predict_y.cpu().detach().numpy()
        predict_ys.append(batch_predict_y)

    predict_ys = np.array(predict_ys)
    shape = predict_ys.shape
    print("predict_ys.shape", predict_ys.shape)
    predict_ys = np.reshape(
        predict_ys, (shape[0]*shape[1], shape[2], shape[3], shape[4]))
    # post process
    predict_ys = post_process(predict_ys)

    # get GT
    gt_ys = y_val

    if i == 0 and False:
        show_pic(x_val[0], gt_ys[0], predict_ys[0])

    # calc jaccard score
    for j in range(len(predict_ys)):
        j_score = calc_jaccard(predict_ys[j], gt_ys[j])
        j_scores.append(j_score)
    print("j_scores:", np.array(j_scores))
    j_score = np.mean(j_scores)
    return j_score


def post_process(batch_predict_y):
    """post process of the result"""
    # shape: [batch_size, 2, width, height]

    batch_predict_y = batch_predict_y[:, 1, :, :]

    res = []
    for predict_y in batch_predict_y:
        # binarization
        predict_y[predict_y > 0] = 1
        predict_y[predict_y <= 0] = 0

        # open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        predict_y = cv2.erode(predict_y, kernel)  # 腐蚀
        predict_y = cv2.dilate(predict_y, kernel)  # 膨胀

        # parse
        predict_y = predict_y.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            predict_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 寻找连通域

        areas = [cv2.contourArea(cnt) for cnt in contours]
        cellIndexs = np.argsort(areas)

        predict_y = np.zeros([predict_y.shape[0], predict_y.shape[1]])
        for j in range(len(cellIndexs)):
            cv2.drawContours(predict_y, contours, j, j, cv2.FILLED)

        predict_y = predict_y.astype(int)
        res.append(predict_y)

    return res


def calc_jaccard(imgA, imgB):
    """calculate the jaccard score"""
    num_A = len(np.unique(imgA))
    num_B = len(np.unique(imgB))

    if num_A < num_B:
        i = imgA
        imgA = imgB
        imgB = i

    unqA = np.unique(imgA)
    for i in range(len(unqA)):
        imgA[imgA == unqA[i]] = i

    unqB = np.unique(imgB)
    for i in range(len(unqB)):
        imgB[imgB == unqB[i]] = i

    hit_matrix = np.zeros([unqA.size, unqB.size])

    for i in range(1, unqA.size):
        A_chan = (imgA == i)
        for j in range(1, unqB.size):
            B_chan = (imgB == j)
            A_and_B = A_chan * B_chan
            B_chan[A_chan == 1] = 1
            hit_matrix[i, j] = np.sum(A_and_B) / np.sum(B_chan)

    jaccard_list = []
    for j in range(1, unqB.size):
        jac_col = np.max(hit_matrix[:, j])
        jaccard_list.append(jac_col)

    j_score = np.sum(jaccard_list) / max(num_A, num_B)
    return j_score


def show_pic(picA, picB, picC):
    plt.subplot(1, 3, 1)
    plt.title("x")
    plt.imshow(picA, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("GT")
    plt.imshow(picB)

    if picC is not None:
        plt.subplot(1, 3, 3)
        plt.title("Predict")
        plt.imshow(picC)

    plt.show()
