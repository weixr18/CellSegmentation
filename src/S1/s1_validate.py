# validate
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

    def validate(self, SHOW_PIC=False):
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

            # post process
            if use_cuda:
                b_predict_y = b_predict_y.cpu().detach().numpy()
                b_predict_y = self.post_process(b_predict_y)
                b_predict_y = b_predict_y.cuda()

            # calc jaccard score
            for j in range(len(b_predict_y)):
                j_score = self.calc_jaccard(
                    b_predict_y[j], b_val_y[j], use_cuda=self.use_cuda)
                j_scores.append(j_score)

                if SHOW_PIC and j_score < 0.5:
                    b_val_x = b_val_x.cpu().detach().numpy()
                    b_predict_y = b_predict_y.cpu().detach().numpy()
                    self.show_pic(b_val_x[j][0], b_val_y[j], b_predict_y[j],
                                  comment=("pic_num: %d, j_score: %f\n" % (i, j_score)))

        print("j_scores_final:", np.array(j_scores))
        j_score = np.mean(j_scores)
        return j_score

    def post_process(self, batch_predict_y, KERNEL_SIZE=(6, 6)):
        """Post process the result."""
        # shape: [batch_size, 2, width, height]

        batch_predict_y = batch_predict_y[:, 1, :, :]

        res = []
        for predict_y in batch_predict_y:
            # binarization
            predict_y[predict_y > 0] = 1
            predict_y[predict_y <= 0] = 0

            # open
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
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

        res = torch.Tensor(res)
        return res

    def calc_jaccard(self, imgA, imgB, use_cuda=True):
        """Calculate the jaccard score"""
        """All this may occur in GPU."""

        unqA = torch.unique(imgA)
        unqB = torch.unique(imgB)
        num_A = len(unqA)
        num_B = len(unqB)

        if num_A < num_B:
            imgA, imgB = imgB, imgA
            num_A, num_B = num_B, num_A
            unqA, unqB = unqB, unqA

        for i in range(num_A):
            imgA[imgA == unqA[i]] = i
        for i in range(num_B):
            imgB[imgB == unqB[i]] = i

        hit_matrix = np.zeros([num_A, num_B])

        if use_cuda:
            for i in range(2, num_A):
                A_chan = (imgA == i).cuda()
                for j in range(1, num_B):
                    B_chan = (imgB == j).cuda()
                    A_and_B = torch.mul(A_chan, B_chan)
                    B_chan[A_chan == 1] = 1
                    hit_matrix[i, j] = torch.sum(
                        A_and_B).float() / torch.sum(B_chan).float()
        else:
            for i in range(2, num_A):
                A_chan = (imgA == i)
                for j in range(1, num_B):
                    B_chan = (imgB == j)
                    A_and_B = torch.mul(A_chan, B_chan)
                    B_chan[A_chan == 1] = 1
                    hit_matrix[i, j] = torch.sum(
                        A_and_B).float() / torch.sum(B_chan).float()

        jaccard_list = []
        for j in range(1, num_B):
            jac_col = np.max(hit_matrix[:, j])
            if jac_col > 0.5:
                jaccard_list.append(jac_col)
            else:
                jaccard_list.append(0)

        j_score = np.sum(jaccard_list) / max(num_A, num_B)
        return j_score

    def show_pic(self, picA, picB, picC=None,
                 A_gray=True, comment=""):
        plt.subplot(1, 3, 1)
        plt.title("x")
        if A_gray:
            plt.imshow(picA, cmap='gray')
        else:
            plt.imshow(picA)

        plt.subplot(1, 3, 2)
        plt.title("GT")
        plt.imshow(picB)

        if picC is not None:
            plt.subplot(1, 3, 3)
            plt.title("Predict")
            plt.imshow(picC)

        if comment is not "":
            plt.text(0, 1, comment, fontsize=14)

        plt.show()
