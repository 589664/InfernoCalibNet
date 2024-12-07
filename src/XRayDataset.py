import torch
import pandas as pd
from config import MEAN, STD
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

        # Read, process, and optionally save the dataframe
        self.dataframe = self._prepare_dataframe()
        if self.output_csv_path:
            self.dataframe.to_csv(self.output_csv_path, index=False)

        # Transformation to be applied (resize, convert to tensor, normalize)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Calculate pos_weight for use with BCEWithLogitsLoss
        self.pos_weight = self._calculate_pos_weight()

    def _prepare_dataframe(self) -> pd.DataFrame:
        # Load CSV file into DataFrame
        df = pd.read_csv(self.csv_file_path)

        # Keep only relevant columns and rename them
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

        # Encode Patient Gender into binary
        df["PG"] = df["PG"].map({"M": 1, "F": 0})

        # Encode View Position into binary
        df["VP"] = df["VP"].map({"PA": 1, "AP": 0})

        # Normalize Patient Age
        df["PA"] = (df["PA"] - df["PA"].min()) / (df["PA"].max() - df["PA"].min())

        # Split Finding Labels into multi-hot encoded list
        unique_labels = list(
            set(label for labels in df["DL"].str.split("|") for label in labels)
        )
        df["MultiHotLabels"] = df["DL"].apply(
            lambda x: [1 if label in x.split("|") else 0 for label in unique_labels]
        )

        self.unique_labels = unique_labels  # Store unique labels for reference
        return df

    def _calculate_pos_weight(self) -> torch.Tensor:
        # Calculate the number of positive samples for each class
        label_sums = (
            self.dataframe["MultiHotLabels"].apply(pd.Series).sum(axis=0).values
        )
        total_samples = len(self.dataframe)

        # Calculate pos_weight for each class
        pos_weight = (total_samples - label_sums) / (
            label_sums + 1e-6
        )  # Avoid division by zero

        # Convert to torch tensor for use in BCEWithLogitsLoss
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

        return pos_weight_tensor

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the image ID directly from the dataframe
        image_id = self.dataframe.iloc[idx]["ImageID"]

        # Load image using the helper method
        image = load_image(image_id)

        # Apply the transformations
        if self.transform:
            image = self.transform(image)

        # Get the multi-hot encoded labels and convert to tensor
        labels = self.dataframe.iloc[idx]["MultiHotLabels"]
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels


# Example usage:
# csv_file_path = "data/metadata.csv"
# output_csv_path = "data/output_metadata.csv"
# dataset = XrayDataset(csv_file_path, output_csv_path)
#
# # Get pos_weight tensor
# pos_weight = dataset.pos_weight
# print("pos_weight:", pos_weight)
#
# # Define loss function with pos_weight
# loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#
# # Get the first data sample
# image, labels = dataset[0]
# predictions = torch.randn(labels.shape, dtype=torch.float32)  # Example prediction tensor
# loss = loss_fn(predictions, labels)
# print("Calculated Loss:", loss.item())
