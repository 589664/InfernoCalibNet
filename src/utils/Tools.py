import os
from tqdm import tqdm
from typing import Tuple
from PIL import Image
from torchvision import transforms

# torch
import torch
from torch.utils.data import DataLoader, Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir)
            for file in files
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label since we're only interested in the images


def compute_mean_std_rgb_dataset(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    device: torch.device = None,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        ImageFolderDataset(root_dir, transform=transforms.ToTensor()),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    total_pixels = 0

    progress_bar = tqdm(loader, desc="Computing mean and std", unit="batch")
    for images, _ in progress_bar:
        images = (
            images.to(device).float().view(images.size(0), 3, -1)
        )  # Flatten to (B, 3, H*W)
        batch_pixels = images.size(0) * images.size(2)
        batch_mean = images.mean(dim=[0, 2])
        batch_std = images.std(dim=[0, 2])

        mean = (mean * total_pixels + batch_mean * batch_pixels) / (
            total_pixels + batch_pixels
        )
        std = (std * total_pixels + batch_std * batch_pixels) / (
            total_pixels + batch_pixels
        )

        total_pixels += batch_pixels

    # Log the final mean and std values to tqdm
    tqdm.write(
        f"Final Mean - R: {mean[0].item():.4f}, G: {mean[1].item():.4f}, B: {mean[2].item():.4f}"
    )
    tqdm.write(
        f"Final Std - R: {std[0].item():.4f}, G: {std[1].item():.4f}, B: {std[2].item():.4f}"
    )

    return (
        (mean[0].item(), std[0].item()),
        (mean[1].item(), std[1].item()),
        (mean[2].item(), std[2].item()),
    )
