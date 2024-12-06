import torch
import pandas as pd
from PIL import Image
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
        self.csv_file_path: str = csv_file_path
        self.output_csv_path: str = output_csv_path
        self.mean: tuple[float, float, float] = mean
        self.std: tuple[float, float, float] = std

        # Read, process and optionally save the dataframe
        self.dataframe: pd.DataFrame = self._prepare_dataframe()
        if self.output_csv_path:
            self.dataframe.to_csv(self.output_csv_path, index=False)

        # Transformation to be applied (resize, convert to tensor, normalize)
        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

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

        return df

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the image ID directly from the dataframe
        image_id: str = self.dataframe.iloc[idx]["ImageID"]

        # Load image using the helper method
        image: Image.Image = load_image(image_id)

        # Convert to numpy array and apply the transformations
        if self.transform:
            image: torch.Tensor = self.transform(image)

        # Get the multi-hot encoded labels (converted here to a tensor)
        labels = self.dataframe.iloc[idx]["MultiHotLabels"]
        labels: torch.Tensor = torch.tensor(labels, dtype=torch.float32)

        return image, labels


# Example usage:
# csv_file_path = "data/metadata.csv"
# output_csv_path = "data/output_metadata.csv"
# dataset = XRayDataset(csv_file_path, output_csv_path)
#
# # Get the first data sample
# image, labels = dataset[0]
# print(image.shape, labels)
