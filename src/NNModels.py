import torch
import torch.nn as nn
from torchvision.models import (
    resnet152,
    ResNet152_Weights,
)
from config import NUM_CL, DROP_RATE


class XrayResNet(nn.Module):
    def __init__(self) -> None:
        super(XrayResNet, self).__init__()

        # Load pre-trained ResNet152 model
        self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

        # Modify the input layer for grayscale (1-channel) images
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Extract the input features of the fully connected layer
        in_features = self.resnet.fc.in_features

        # Replace the fully connected layer with a custom one for the desired number of classes, adding Dropout
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=DROP_RATE),  # Dropout with configurable probability
            nn.Linear(in_features, NUM_CL),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
