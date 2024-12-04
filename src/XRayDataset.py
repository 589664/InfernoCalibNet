import torch
import pandas as pd
from PIL import Image
from .utils.Tools import load_image
from torchvision import transforms
from torch.utils.data import Dataset


class XRayDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        img_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ):
        self.dataframe: pd.DataFrame = dataframe
        self.image_dir: str = image_dir
        self.img_size: int = img_size
        self.mean: tuple[float, float, float] = mean
        self.std: tuple[float, float, float] = std

        # Transformation to be applied (resize, convert to tensor, normalize)
        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean, std=self.std
                ),  # Use computed mean/std for RGB
            ]
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the full image path directly from the dataframe
        img_path: str = self.dataframe.iloc[idx]["ImagePath"]

        # Load image using the helper method
        image: Image.Image = load_image(img_path, self.img_size)

        # Convert to numpy array and apply the transformations
        if self.transform:
            image: torch.Tensor = self.transform(image)

        # Get the multi-hot encoded labels
        labels: torch.Tensor = torch.tensor(
            self.dataframe.iloc[idx]["MultiHotLabels"], dtype=torch.float32
        )

        return image, labels
