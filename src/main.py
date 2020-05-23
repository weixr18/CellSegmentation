from train import Trainer


if __name__ == "__main__":

    hyperparameters = {
        "batch_size": 1,
        "learning_rate": 1e-4,
    }

    data_dir = ""
    mask_dir = ""
    valid_rate = 0.1
    save_path = ""

    trainer = Trainer()
    trainer.setup(data_dir, mask_dir, valid_rate)

    trainer.run()
    trainer.save_module(save_path)

    pass
