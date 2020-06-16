import argparse

from test import Tester
from train import Trainer
from predict import Predictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="Train")
    parser.add_argument("--stage", "-s", default="S1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    mode = args.mode

    if mode == "Train":
        hyper_parameters = {
            "batch_size": 1,
            "learning_rate": 1e-4,
            "threads": 0,
            "epochs": 2000,
            "epoch_lapse": 100000,
            "epoch_save": 40,
            "TTA_KERNEL_SIZE": (6, 6),
            "BG_KERNEL_SIZE": (8, 8),
            "DILATE_ITERATIONS": 20,
            "BIN_THRESHOLD": 0.6,
            "input_size": (500, 500),
        }

        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/train/"
        mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/train_GT/SEG/"
        module_save_dir = "D:/Machine_Learning/Codes/CellSegment/save/"
        tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/_tmp/"

        model_path = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200614223438epoch-240.pth"
        valid_rate = 0.1
        use_cuda = True
        use_exist_dataset = True

        trainer = Trainer()
        trainer.setup(cell_dir=cell_dir,
                      mask_dir=mask_dir,
                      module_save_dir=module_save_dir,
                      tmp_dir=tmp_dir,
                      model_path=model_path,
                      valid_rate=valid_rate,
                      hyper_params=hyper_parameters,
                      use_cuda=use_cuda,
                      FREEZE_PARAM=True,
                      use_exist_dataset=use_exist_dataset,
                      PRETRAINED=True)

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
            cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/train/"
            mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/train_GT/SEG/"
            tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/_tmp/_test/"
            exist_res_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/test_RES"
            test_rate = 0.1
            use_exist_dataset = True

            USE_EXIST_RES = False
            use_cuda = True
            TTA = False
            SHOW_PIC = True

        for e in range(50, 100, 50):

            module_path = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200614215315epoch-20.pth"

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
        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/test/"
        model_path = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200614225110epoch-360.pth"
        save_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset2/test_RES/"
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
