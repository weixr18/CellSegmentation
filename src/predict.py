import cv2
import torch
from torch.utils.data import DataLoader

from unet import UNet
from data import TestSet
from validate import Validator


SHOW_PIC = True


class Predictor():

    def __init__(self, module_path, cell_dir, save_dir,
                 hyper_params, use_cuda):

        self.test_set = TestSet(cell_dir)
        print("test number:", len(self.dataset))

        self.hyper_params = hyper_params
        self.data_loader = DataLoader(
            dataset=self.dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )
        self.save_dir = save_dir
        self.use_cuda = use_cuda

        self.unet = UNet(n_channels=1, n_classes=2,)
        self.unet.load_state_dict(torch.load(module_path))
        if use_cuda:
            self.unet = self.unet.cuda()

    pass

    def predict(self, TTA=True):

        width_out = 628
        height_out = 628
        batch_size = self.hyper_params["batch_size"]
        use_cuda = self.use_cuda

        predict_ys = []
        for i, data in enumerate(self.data_loader):

            """preprocess"""
            b_val_x, b_val_y = data

            """Test time augmentation"""
            # S b_val_x: [batch_size, width, height]
            if TTA:
                b_val_x_fh = torch.flip(b_val_x, dims=[1])
                b_val_x_fv = torch.flip(b_val_x, dims=[2])
                b_val_x_90 = torch.rot90(b_val_x, 1, dims=(1, 2))
                b_val_x_180 = torch.rot90(b_val_x, 2, dims=(1, 2))
                b_val_x_270 = torch.rot90(b_val_x, 3, dims=(1, 2))

                b_val_x_list = [
                    b_val_x,
                    b_val_x_fh,
                    b_val_x_fv,
                    b_val_x_90,
                    b_val_x_180,
                    b_val_x_270,
                ]
            else:
                b_val_x_list = [b_val_x]

            """get binary output"""
            # S b_val_x_list: [6 or 1, batch_size, width, height]

            b_y_list_cpu = []
            for b_x in b_val_x_list:

                # S b_x: [batch_size, width, height]
                if not isinstance(b_x, torch.Tensor):
                    b_x = T.ToTensor()(b_x)
                if (len(b_x.size()) == 3):
                    b_x = b_x.unsqueeze(1)
                elif (len(b_x.size() == 2)):
                    b_x = b_x.unsqueeze(0)
                    b_x = b_x.unsqueeze(1)

                # S b_x: [batch_size, 1, width, height]
                if use_cuda:
                    b_x = b_x.cuda()

                """get raw output"""
                b_predict_y = self.unet(b_x)

                """binarization"""
                # S b_predict_y: [batch_size, 2, width, height]
                b_predict_y_cpu = self.binarization(b_predict_y).detach().cpu()

                # S b_predict_y: [batch_size, width, height]
                b_y_list_cpu.append(b_predict_y_cpu)

            """Augmentation vote"""
            # S b_y_list_cpu: [6 or 1, batch_size, width, height]

            if TTA:
                # S b_y_list_cpu[n]: [batch_size, width, height]
                b_y_list_cpu[1] = torch.flip(b_y_list_cpu[1], dims=[1])
                b_y_list_cpu[2] = torch.flip(b_y_list_cpu[2], dims=[2])
                b_y_list_cpu[3] = torch.rot90(b_y_list_cpu[3], 3, dims=(1, 2))
                b_y_list_cpu[4] = torch.rot90(b_y_list_cpu[4], 2, dims=(1, 2))
                b_y_list_cpu[5] = torch.rot90(b_y_list_cpu[5], 1, dims=(1, 2))

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

                """Open operation"""
                for n in range(6):
                    b_y_list_cpu[n] = b_y_list_cpu[n].numpy()
                    for j in range(batch_size):
                        b_y_list_cpu[n][j] = cv2.morphologyEx(
                            src=b_y_list_cpu[n][j],
                            op=cv2.MORPH_OPEN,
                            kernel=kernel)
                    b_y_list_cpu[n] = torch.tensor(b_y_list_cpu[n])
                    b_y_list_cpu[n] = b_y_list_cpu[n].unsqueeze(0)

                """Vote"""
                # S b_y_list_cpu[n]: [1, batch_size, width, height]
                b_predict_y = torch.cat(tuple(b_y_list_cpu), dim=0)
                b_predict_y = torch.mean(b_predict_y, dim=0)

                # S b_predict_y: [batch_size, width, height]
                b_predict_y[b_predict_y > 0.5] = 1
                b_predict_y[b_predict_y <= 0.5] = 0

            else:
                b_predict_y = b_y_list_cpu[0]

            """Instance Sparse"""
            # S b_predict_y: [batch_size, width, height]
            b_predict_y = self.instance_sparse(b_predict_y)

            for y in b_predict_y:
                y = y.detach().cpu()
                predict_ys.append(y)

            pass

        self.save_pics(predict_ys)
        pass

    def save_pics(self, predict_ys):

        save_dir = self.save_dir
        pass

    def binarization(self, batch_predict_y):
        # S b_predict_y: [batch_size, 2, width, height]

        # sqeeze
        batch_predict_y_1 = torch.softmax(batch_predict_y, dim=1)
        batch_predict_y_1 = batch_predict_y_1[:, 1, :, :]
        THRESHOLD_1 = 0.5
        batch_predict_y_1[batch_predict_y_1 > THRESHOLD_1] = 1
        batch_predict_y_1[batch_predict_y_1 <= THRESHOLD_1] = 0

        batch_predict_y = batch_predict_y_1
        # S b_predict_y: [batch_size, width, height]

        return batch_predict_y

    def instance_sparse(self, batch_predict_y, KERNEL_SIZE=(6, 6)):
        """Post process the result."""
        # TODO: use the watershed algorithm.

        # shape: [batch_size, width, height]

        res = []
        for predict_y in batch_predict_y:

            predict_y = predict_y.numpy().astype(np.uint8) * 255
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

    def show_pic(self, picA, picB, picC=None,
                 is_gray=(True, False, False), comment=""):
        plt.subplot(1, 3, 1)
        plt.title("x")
        if is_gray[0]:
            plt.imshow(picA, cmap='gray')
        else:
            plt.imshow(picA)

        plt.subplot(1, 3, 2)
        plt.title("GT")
        if is_gray[1]:
            plt.imshow(picB, cmap='gray')
        else:
            plt.imshow(picB)

        if picC is not None:
            plt.subplot(1, 3, 3)
            plt.title("Predict")
            if is_gray[2]:
                plt.imshow(picC, cmap='gray')
            else:
                plt.imshow(picC)

        if comment is not "":
            plt.text(0, 1, comment, fontsize=14)

        plt.show()
