import torch
import pandas as pd
from config import MEAN, STD, NUM_CL
from torchvision import transforms
from torch.utils.data import Dataset
from .utils.Tools import load_image


class XrayDataset(Dataset):
    def __init__(
        self,
        csv_file_path: str,
        output_csv_path: str = None,
        mean: tuple[float, float, float] = MEAN,
        std: tuple[float, float, float] = STD,
    ):
        self.csv_file_path = csv_file_path
        self.output_csv_path = output_csv_path
        self.mean = mean
        self.std = std

        self.dataframe = self._prepare_dataframe()
        if self.output_csv_path:
            self.dataframe.to_csv(self.output_csv_path, index=False)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.pos_weight = self._calculate_pos_weight()
        assert len(self.unique_labels) == NUM_CL

    def _prepare_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_file_path)

        df = df[
            [
                "Image Index",
                "Finding Labels",
                "Patient Age",
                "Patient Gender",
                "View Position",
            ]
        ]
        df.columns = ["ImageID", "DL", "PA", "PG", "VP"]

        df["PG"] = df["PG"].map({"M": 1, "F": 0})
        df["VP"] = df["VP"].map({"PA": 1, "AP": 0})
        df["PA"] = (df["PA"] - df["PA"].min()) / (df["PA"].max() - df["PA"].min())

        all_labels = set(
            label for labels in df["DL"].str.split("|") for label in labels
        )
        all_labels.discard("No Finding")
        unique_labels = sorted(list(all_labels))

        def encode_labels(label_string):
            labels = label_string.split("|")
            if "No Finding" in labels:
                return [0] * len(unique_labels)
            return [1 if label in labels else 0 for label in unique_labels]

        df["MultiHotLabels"] = df["DL"].apply(encode_labels)

        self.unique_labels = unique_labels
        return df

    def _calculate_pos_weight(self) -> torch.Tensor:
        label_sums = (
            self.dataframe["MultiHotLabels"].apply(pd.Series).sum(axis=0).values
        )
        total_samples = len(self.dataframe)

        pos_weight = (total_samples - label_sums) / (label_sums + 1e-6)
        return torch.tensor(pos_weight, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.dataframe.iloc[idx]["ImageID"]

        image = load_image(image_id)
        if self.transform:
            image = self.transform(image)

        labels = self.dataframe.iloc[idx]["MultiHotLabels"]
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels


# Example usage:
# csv_file_path = "data/metadata.csv"
# output_csv_path = "data/output_metadata.csv"
# dataset = XrayDataset(csv_file_path, output_csv_path)
#
# pos_weight = dataset.pos_weight
# print("pos_weight:", pos_weight)
#
# loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#
# image, labels = dataset[0]
# predictions = torch.randn(labels.shape, dtype=torch.float32)
# loss = loss_fn(predictions, labels)
# print("Calculated Loss:", loss.item())
