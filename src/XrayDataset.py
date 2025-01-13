import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config import IMG_SIZE, OUT_DIR, MEAN, STD


class XrayDataset(Dataset):
    def __init__(self, split: str, augmentations: bool = True) -> None:
        csv_path = os.path.join(OUT_DIR, f"{split}.csv")
        self.metadata = pd.read_csv(csv_path)

        # Define augmentations for training and simple transformations for validation/test
        if augmentations:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[MEAN], std=[STD]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[MEAN], std=[STD]),
                ]
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError(
                f"Index {idx} out of range for dataset with length {len(self.metadata)}"
            )

        row = self.metadata.iloc[idx]
        image_path = row["ImagePath"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        # Parse the MultiHotLabels correctly
        label = torch.tensor(eval(row["MultiHotLabels"]), dtype=torch.float)
        return image, label


# Example usage:
# train_dataset = XrayDataset(split="train", augmentations=True)
# validate_dataset = XrayDataset(split="validate", augmentations=False)
# test_dataset = XrayDataset(split="test", augmentations=False)
