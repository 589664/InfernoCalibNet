import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class InfernoCalibNet(nn.Module):
    def __init__(self, num_classes=3, drop_rate=0.6):
        super(InfernoCalibNet, self).__init__()

        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Modify conv1 to accept 1-channel (grayscale) images with optimized parameters
        base_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Retain all layers except the final FC and AvgPool
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])

        # Custom classifier
        num_feat = base_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Conv2d(num_feat, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),  # Multiclass classification logits
        )

    def forward(self, x):
        x = self.base_model(x)  # Extract features from base ResNet
        x = self.classifier(x)  # Custom classification layers
        return x  # Output raw logits


# Example usage
# model = XrayResNet(num_classes=3)
# print(model)
