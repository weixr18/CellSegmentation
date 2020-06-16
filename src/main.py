import argparse

from test import Tester
from train import Trainer
from predict import Predictor


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
            "input_size": (628, 628),
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
                      use_cuda=use_cuda,
                      PRETRAINED=False)

        trainer.train()
        trainer.save_module()

    elif mode == "Test":

        hyper_parameters = {
            "batch_size": 1,
            "threads": 0,
            "TTA_KERNEL_SIZE": (6, 6),
            "BG_KERNEL_SIZE": (8, 8),
            "DILATE_ITERATIONS": 10,
            "BIN_THRESHOLD": 0.60,
        }

        DEMO = False  # DEBUG

        if DEMO:
            cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_demo/"
            mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_demo_mask/"
            tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_tmp/_test/"
            test_rate = 1
            use_exist_dataset = False

        else:
            cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train/"
            mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/train_GT/SEG/"
            tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_tmp/_test/"
            exist_res_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/test_RES"
            USE_EXIST_RES = False
            test_rate = 0.1
            use_exist_dataset = True

            use_cuda = True
            TTA = False
            SHOW_PIC = False

        for e in range(50, 100, 50):

            module_path = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200524epoch-2000.pth"

            tester = Tester(
                module_path=module_path,
                cell_dir=cell_dir,
                mask_dir=mask_dir,
                tmp_dir=tmp_dir,
                exist_res_dir=exist_res_dir,
                hyper_params=hyper_parameters,
                use_cuda=use_cuda,
                use_exist_dataset=use_exist_dataset,
                test_rate=test_rate,
                USE_EXIST_RES=USE_EXIST_RES
            )
            import time
            tic = time.time()
            test_acc = tester.test(
                SHOW_PIC=SHOW_PIC,
                TTA=TTA
            )
            toc = time.time()
            print("module:", module_path.split('/')[-1])
            print("test accuracy:", test_acc, "time:", toc - tic)

    elif mode == "Predict":

        hyper_parameters = {
            "batch_size": 1,
            "threads": 0,
            "TTA_KERNEL_SIZE": (6, 6),
            "BG_KERNEL_SIZE": (8, 8),
            "DILATE_ITERATIONS": 10,
            "BIN_THRESHOLD": 0.6,
        }
        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/test/"
        model_path = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200524epoch-2000.pth"
        save_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/test_RES/"
        use_cuda = True
        TTA = False

        predictor = Predictor(model_path=model_path,
                              cell_dir=cell_dir,
                              save_dir=save_dir,
                              hyper_params=hyper_parameters,
                              use_cuda=use_cuda
                              )

        predictor.predict(TTA=TTA)
        pass

    elif mode == "Validate":

        hyper_parameters = {
            "batch_size": 1,
            "threads": 0,
        }

        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/test/"
        mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/test_RES"
        tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_tmp/_test/"
        exist_res_dir = ""
        use_exist_dataset = False
        test_rate = 0
        USE_EXIST_RES = True

        module_path = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200524epoch-2000.pth"
        use_cuda = True
        SHOW_PIC = True
        TTA = False

        tester = Tester(
            module_path=module_path,
            cell_dir=cell_dir,
            mask_dir=mask_dir,
            tmp_dir=tmp_dir,
            exist_res_dir=exist_res_dir,
            hyper_params=hyper_parameters,
            use_cuda=use_cuda,
            use_exist_dataset=use_exist_dataset,
            test_rate=test_rate,
            USE_EXIST_RES=USE_EXIST_RES
        )
        import time
        tic = time.time()
        test_acc = tester.test(
            SHOW_PIC=SHOW_PIC,
            TTA=TTA
        )
        toc = time.time()
        print("module:", module_path.split('/')[-1])
        print("test accuracy:", test_acc, "time:", toc - tic)

    pass
