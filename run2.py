# from src.XrayDataset import XRayDataset
from src.utils.Tools import split_and_save_dataframe, count_images_per_class

from config import ROOT_DIR


def main():

    split_and_save_dataframe()

    count_images_per_class(txt_path=ROOT_DIR / "data" / "raw" / "train_val_list.txt")


if __name__ == "__main__":
    main()
