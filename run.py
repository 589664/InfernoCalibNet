from config import OUT_DIR
from src.XrayDataset import XrayDataset

from src.ICNTrainer import ICNTrainer
from src.NNModels import DenseNet201

from src.utils.Tools import split_and_save_dataframe


def main():

    split_and_save_dataframe(reductRate=0.3)

    # trainDS = XrayDataset(csv_file_path=OUT_DIR / "train.csv")
    # valDS = XrayDataset(csv_file_path=OUT_DIR / "val.csv")

    # pos_weight = trainDS.pos_weight
    # trainer = ICNTrainer(DenseNet201, trainDS, valDS, class_weights=pos_weight)
    # trainer.fit()


if __name__ == "__main__":
    main()
