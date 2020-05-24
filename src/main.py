import argparse
from S1.s1_main import s1_main


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="Train")
    parser.add_argument("--stage", "-s", default="S1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    mode = args.mode
    stage = args.stage

    if stage == "S1":
        s1_main(mode)
    pass
