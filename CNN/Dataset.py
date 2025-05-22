import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

#=======================================================================================================================
# 🚀 Main Code
#=======================================================================================================================
class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, transform=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Augmentation for training
        self.augmentation_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.RandomResizedCrop(size=(256, 256), scale=(0.75, 1.0)),
            ]
        )

        # Standard preprocessing for all images
        self.base_transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.49765], std=[0.22854]),
                # Method 1 - Average mean: 0.49765, Average std: 0.22854
                # Method 2 - Mean: 0.49765, Std: 0.24790
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["IMGPATH"]
        image = Image.open(img_path).convert("L")
        label = torch.tensor(eval(self.data.iloc[idx]["MULTIHOT"]), dtype=torch.float32)

        # Apply augmentation only if transform is enabled
        if self.transform:
            image = self.augmentation_transform(image)

        # Apply base transformation to all images
        image = self.base_transform(image)

        return image, label