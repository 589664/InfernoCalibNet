import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


# Custom Dataset Class
class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, transform=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Augmentation for training
        self.augmentation_transform = T.Compose(
            [
                T.RandomRotation(degrees=13),
                T.RandomAffine(degrees=0, translate=(0.06, 0.06)),
                T.RandomResizedCrop(size=(256, 256), scale=(0.75, 1.0)),
            ]
        )

        # Standard preprocessing for all images
        self.base_transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485], std=[0.229]
                ),  # Normalize using standard ImageNet values
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["IMGPATH"]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        label = torch.tensor(self.data.iloc[idx]["CLASS"], dtype=torch.long)

        # Apply augmentation only if transform is enabled (assumed to be training set)
        if self.transform:
            image = self.augmentation_transform(image)

        # Apply base transformation to all images
        image = self.base_transform(image)

        return image, label
