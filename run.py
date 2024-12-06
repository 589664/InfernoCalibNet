from config import OUT_DIR
from src.XrayDataset import XrayDataset
from src.ICNTrainer import ICNTrainer
from src.NNModels import DenseNet201

# from src.utils.Tools import split_and_save_dataframe


def main():

    # split_and_save_dataframe()

    trainDS = XrayDataset(csv_file_path=OUT_DIR / "train.csv")
    valDS = XrayDataset(csv_file_path=OUT_DIR / "val.csv")

    trainer = ICNTrainer(DenseNet201, trainDS, valDS)
    trainer.fit()


if __name__ == "__main__":
    main()
