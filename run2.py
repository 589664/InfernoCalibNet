from config import OUT_DIR
from src.XrayDataset import XRayDataset

# from src.utils.Tools import split_and_save_dataframe


def main():

    # split_and_save_dataframe()

    trainDS = XRayDataset(csv_file_path=OUT_DIR / "train.csv")
    valDS = XRayDataset(csv_file_path=OUT_DIR / "val.csv")


if __name__ == "__main__":
    main()
