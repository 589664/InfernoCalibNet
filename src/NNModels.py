import torch
import torch.nn as nn
from torchvision.models import (
    resnet152,
    resnet50,
    ResNet152_Weights,
    ResNet50_Weights,
)
from config import NUM_CL, DROP_RATE


class XrayResNet(nn.Module):
    def __init__(self, model_type="resnet152") -> None:
        super(XrayResNet, self).__init__()

        if model_type == "resnet152":
            self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        elif model_type == "resnet50":
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Invalid model_type. Choose 'resnet152' or 'resnet50'.")

        # Modify the input layer for grayscale (1-channel) images
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Add Dropout after layer4
        self.resnet.layer4 = nn.Sequential(self.resnet.layer4, nn.Dropout(p=DROP_RATE))

        # Add Dropout after global average pooling
        self.resnet.avgpool = nn.Sequential(
            self.resnet.avgpool, nn.Dropout(p=DROP_RATE)
        )

        # Replace the fully connected layer with Dropout and a custom output layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=DROP_RATE),  # Dropout before FC
            nn.Linear(in_features, NUM_CL),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


# Example usage:
# model = XrayResNet(model_type="resnet152")
# model = XrayResNet(model_type="resnet50")
