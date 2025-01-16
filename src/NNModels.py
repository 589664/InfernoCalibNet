import torch.nn as nn
from torchvision.models import (
    resnet152,
    resnet101,
    resnet50,
)
from config import NUM_CL, DROP_RATE, IMG_SIZE

import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet152


class XrayResNet(nn.Module):
    def __init__(self, model_type="resnet50"):
        super(XrayResNet, self).__init__()

        # Select ResNet architecture based on model_type
        if model_type == "resnet50":
            base_model = resnet50(weights=None)
        elif model_type == "resnet101":
            base_model = resnet101(weights=None)
        elif model_type == "resnet152":
            base_model = resnet152(weights=None)
        else:
            raise ValueError(
                "Invalid model_type. Choose from 'resnet50', 'resnet101', or 'resnet152'."
            )

        # Modify conv1 to accept 1-channel (grayscale) images
        base_model.conv1 = nn.Conv2d(
            in_channels=1,  # Change to 1-channel input
            out_channels=base_model.conv1.out_channels,  # Keep output channels same
            kernel_size=base_model.conv1.kernel_size,
            stride=base_model.conv1.stride,
            padding=base_model.conv1.padding,
            bias=base_model.conv1.bias,
        )

        # Retain all layers except the final FC and AvgPool
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])

        # Add custom layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # GlobalAveragePooling2D equivalent
        self.flatten = nn.Flatten()  # Flatten for fully connected layers
        self.dropout1 = nn.Dropout(p=DROP_RATE)
        self.fc1 = nn.Linear(base_model.fc.in_features, 512)
        self.dropout2 = nn.Dropout(p=DROP_RATE)
        self.fc2 = nn.Linear(512, NUM_CL)  # Final layer with NUM_CL outputs

    def forward(self, x):
        x = self.base_model(x)  # Extract features from base ResNet
        x = self.global_avg_pool(x)  # Global average pooling
        x = self.flatten(x)  # Flatten the pooled features
        x = self.dropout1(x)  # First Dropout
        x = self.fc1(x)  # First fully connected layer
        x = self.dropout2(x)  # Second Dropout
        x = self.fc2(x)  # Output layer
        return x  # Output raw logits


# Example usage:
# model = XrayResNet(model_type="resnet50")


# TRANSFER LEARNING:
# import torch
# import torch.nn as nn
# from torchvision.models import (
#     resnet152,
#     resnet50,
#     ResNet152_Weights,
#     ResNet50_Weights,
# )
# from config import NUM_CL, DROP_RATE


# class XrayResNet(nn.Module):
#     def __init__(self, model_type="resnet152") -> None:
#         super(XrayResNet, self).__init__()

#         if model_type == "resnet152":
#             self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
#         elif model_type == "resnet50":
#             self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         else:
#             raise ValueError("Invalid model_type. Choose 'resnet152' or 'resnet50'.")

#         # Modify the input layer for grayscale (1-channel) images
#         self.resnet.conv1 = nn.Conv2d(
#             in_channels=1,
#             out_channels=self.resnet.conv1.out_channels,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False,
#         )

#         # Add Dropout after layer4
#         self.resnet.layer4 = nn.Sequential(self.resnet.layer4, nn.Dropout(p=DROP_RATE))

#         # Add Dropout after global average pooling
#         self.resnet.avgpool = nn.Sequential(
#             self.resnet.avgpool, nn.Dropout(p=DROP_RATE)
#         )

#         # Replace the fully connected layer with Dropout and a custom output layer
#         in_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Sequential(
#             nn.Dropout(p=DROP_RATE),  # Dropout before FC
#             nn.Linear(in_features, NUM_CL),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.resnet(x)


# # Example usage:
# # model = XrayResNet(model_type="resnet152")
# # model = XrayResNet(model_type="resnet50")
