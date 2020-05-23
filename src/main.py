import argparse
from S1.s1_train import Trainer
from S1.s1_test import Tester


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
            "threads": 0,
            "epochs": 1,
            "epoch_lapse": 1,
            "epoch_save": 50,
        }

        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train/"
        mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train_GT/SEG/"
        module_save_dir = "D:/Machine_Learning/Codes/CellSegment/save/"
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

        trainer.train()
        trainer.save_module()

    elif mode == "Test":

        hyper_parameters = {
            "batch_size": 1,
            "threads": 0,
        }

        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train/"
        mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train_GT/SEG/"
        tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_tmp/_test/"
        use_cuda = True

        for e in range(500, 550, 50):

            module_path = "D:/Machine_Learning/Codes/CellSegment/save/"\
                + "unet-20200523epoch-%d.pth" % (e)
            SHOW_PIC = True

            tester = Tester(
                module_path=module_path,
                cell_dir=cell_dir,
                mask_dir=mask_dir,
                tmp_dir=tmp_dir,
                hyper_params=hyper_parameters,
                use_cuda=use_cuda,
                use_exist=True,
                test_rate=0.1,
            )
            import time
            tic = time.time()
            test_acc = tester.test(SHOW_PIC=SHOW_PIC)
            toc = time.time()
            print("epoch:", e, "test accuracy:", test_acc, "time:", toc - tic)

    elif mode == "Predict":
        pass

    pass
