import torch
from torch.utils.data import Dataset, DataLoader

from unet import UNet
from data import get_dataset, get_test_dataset
from validate import Validator


class Tester():

    def __init__(self, module_path, cell_dir, mask_dir, tmp_dir, exist_res_dir,
                 hyper_params, use_cuda, test_rate=1.0, use_exist_dataset=False,
                 USE_EXIST_RES=True,):

        print("Test rate:", test_rate)
        if USE_EXIST_RES:
            self.dataset = get_test_dataset(
                cell_dir=cell_dir,
                GT_dir=mask_dir,
                res_dir=exist_res_dir,
                valid_rate=1 - test_rate,
                tmp_dir=tmp_dir,
                use_exist_dataset=use_exist_dataset,
                for_test=True
            )
        else:
            self.dataset, _ = get_test_dataset(
                cell_dir=cell_dir,
                mask_dir=mask_dir,
                valid_rate=1 - test_rate,
                tmp_dir=tmp_dir,
                use_exist_dataset=use_exist_dataset,
            )

        print("test number:", len(self.dataset))

        self.hyper_params = hyper_params
        self.data_loader = DataLoader(
            dataset=self.dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )

        self.unet = UNet(n_channels=1, n_classes=2,)
        self.unet.load_state_dict(torch.load(module_path))
        if use_cuda:
            self.unet = self.unet.cuda()

        self.v = Validator(unet=self.unet,
                           hyper_params=self.hyper_params,
                           use_cuda=use_cuda,
                           data_loader=self.data_loader,
                           USE_EXIST_RES=USE_EXIST_RES,
                           exist_res_dir=exist_res_dir)

    def test(self, SHOW_PIC=False, TTA=False):
        return self.v.validate(SHOW_PIC=SHOW_PIC, TTA=TTA)
    pass
