import argparse
from train import Trainer
from test import Tester


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="Train")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    mode = args["mode"]

    if mode == "Train":
        hyper_parameters = {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "threads": 2,
            "epochs": 1,
            "epoch_lapse": 1,
        }

        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train/"
        mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train_GT/SEG"
        module_save_dir = "D:/Machine_Learning/Codes/CellSegment/save"
        tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_tmp/"

        valid_rate = 0.1
        use_cuda = True

        trainer = Trainer()
        trainer.setup(cell_dir=cell_dir,
                      mask_dir=mask_dir,
                      module_save_dir=module_save_dir,
                      tmp_dir=tmp_dir,
                      valid_rate=valid_rate,
                      hyper_params=hyper_parameters,
                      use_cuda=use_cuda)

        trainer.run()
        trainer.save_module()

    else if mode == "Test":

        modelPath = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200520185855.pth"

        unet = UNet(n_channels=1, n_classes=2,)
        unet.load_state_dict(torch.load(modelPath))
        if use_gpu:
            unet = unet.cuda()

        tester = Tester()

        val_loss = get_val_loss(x_val, y_val, width_out, height_out,
                                unet, batch_size, use_gpu)

    pass
