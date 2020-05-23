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
    mode = args.mode

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

    elif mode == "Test":

        hyper_parameters = {
            "batch_size": 2,
            "threads": 0,
        }

        module_dir = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200520185855.pth"
        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train/"
        mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train_GT/SEG"
        tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_tmp/_test/"
        use_cuda = False

        tester = Tester(
            module_dir=module_dir,
            cell_dir=cell_dir,
            mask_dir=mask_dir,
            tmp_dir=tmp_dir,
            hyper_params=hyper_parameters,
            use_cuda=use_cuda,
            test_rate=0.1
        )
        import time
        tic = time.time()
        test_acc = tester.test()
        toc = time.time()
        print("test accuracy:", test_acc, "time:", toc-tic)

    pass
