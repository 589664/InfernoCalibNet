import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, transform=True, return_aux=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.return_aux = return_aux

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
                T.Normalize(mean=[0.07753], std=[1.15581]),
                # T.Normalize(mean=[0.485], std=[0.229]), # ImageNET
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["IMGPATH"]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        label = torch.tensor(self.data.iloc[idx]["CLASS"], dtype=torch.long)

        # Extract auxiliary data if return_aux is enabled
        aux_data = (
            {
                key: self.data.iloc[idx][key]
                for key in [
                    "DISEASELABEL",
                    "FOLLOWUP",
                    "PATID",
                    "AGE",
                    "GENDER",
                    "VP",
                    "IMGPATH",
                ]
            }
            if self.return_aux
            else None
        )

        # Apply augmentation only if transform is enabled
        if self.transform:
            image = self.augmentation_transform(image)

        # Apply base transformation to all images
        image = self.base_transform(image)

        if self.return_aux:
            return image, label, aux_data
        return image, label
