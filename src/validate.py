import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy


class Validator():

    def __init__(self, unet,
                 hyper_params,
                 use_cuda,
                 data_loader):
        self.unet = unet
        self.hyper_params = hyper_params
        self.use_cuda = use_cuda
        self.data_loader = data_loader
        pass

    def validate(self):
        width_out = 628
        height_out = 628
        batch_size = self.hyper_params["batch_size"]
        use_cuda = self.use_cuda

        j_scores = []
        for i, data in enumerate(self.data_loader):
            # preprocess
            b_val_x, b_val_y = data
            if (len(b_val_x.size()) == 3):
                b_val_x = b_val_x.unsqueeze(1)
            if use_cuda:
                b_val_x = b_val_x.cuda()

            # get predict
            b_predict_y = self.unet(b_val_x)
            b_predict_y = b_predict_y.cpu().detach().numpy()

            # post process
            b_predict_y = self.post_process(b_predict_y)

            # get GT
            b_gt_y = b_val_y

            if i == 0 and False:
                self.show_pic(b_val_x[0], b_gt_y[0], b_predict_y[0])

            # calc jaccard score
            for j in range(len(b_predict_y)):
                j_score = self.calc_jaccard(b_predict_y[j], b_gt_y[j])
                j_scores.append(j_score)

        print("j_scores:", np.array(j_scores))
        j_score = np.mean(j_scores)
        return j_score

    def post_process(self, batch_predict_y):
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
            if cv2.__version__[0] == '3':
                __, contours, _ = cv2.findContours(
                    predict_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 寻找连通域
            elif cv2.__version__[0] == '4':
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

    def calc_jaccard(self, imgA, imgB):
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

    def show_pic(self, picA, picB, picC):
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
