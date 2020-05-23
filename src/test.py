import torch
from torch.utils.data import Dataset, DataLoader

from unet import UNet
from data import get_dataset
from validate import Validator


class Tester():

    def __init__(self, module_dir, cell_dir, mask_dir, tmp_dir,
                 hyper_params, use_cuda, test_rate=1.0):

        self.dataset, _ = get_dataset(
            cell_dir, mask_dir, 1 - test_rate, tmp_dir)

        print("test number:", len(self.dataset))

        self.hyper_params = hyper_params

        self.data_loader = DataLoader(
            dataset=self.dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )

        self.unet = UNet(n_channels=1, n_classes=2,)
        self.unet.load_state_dict(torch.load(module_dir))
        if use_cuda:
            self.unet = self.unet.cuda()

        self.v = Validator(unet=self.unet,
                           hyper_params=self.hyper_params,
                           use_cuda=use_cuda,
                           data_loader=self.data_loader)

    def test(self):
        return self.v.validate()
    pass
