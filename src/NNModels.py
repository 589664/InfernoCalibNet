import torch
import torch.nn as nn
from torchvision.models import (
    densenet201,
    efficientnet_b4,
    EfficientNet_B4_Weights,
    DenseNet201_Weights,
)
from config import NUM_CLASSES


class DenseNet201(nn.Module):
    def __init__(self) -> None:
        super(DenseNet201, self).__init__()

        # Load pre-trained DenseNet201 model
        self.densenet201 = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)

        # Extract the input features of the classifier layer
        kernel_count = self.densenet201.classifier.in_features

        # Replace the classifier layer with a custom one for the desired number of classes
        self.densenet201.classifier = nn.Sequential(
            nn.Linear(kernel_count, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet201(x)


# Example usage for DenseNet201:
# model = DenseNet201()
# input_tensor = torch.randn(8, 3, 224, 224)  # Batch of 8 RGB images of size 224x224
# output = model(input_tensor)
# print(output.shape)  # Should be [8, NUM_CLASSES] where NUM_CLASSES is the number of classes


class EfficientNetB4(nn.Module):
    def __init__(self) -> None:
        super(EfficientNetB4, self).__init__()

        # Load pre-trained EfficientNet B4 model
        self.efficientnet_b4 = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

        # Extract the input features of the classifier layer
        kernel_count = self.efficientnet_b4.classifier[1].in_features

        # Replace the classifier layer with a custom one for the desired number of classes
        self.efficientnet_b4.classifier = nn.Sequential(
            nn.Linear(kernel_count, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet_b4(x)


# Example usage for EfficientNetB4:
# model = EfficientNetB4()
# input_tensor = torch.randn(8, 3, 380, 380)  # Batch of 8 RGB images of size 224x224
# output = model(input_tensor)
# print(output.shape)  # Should be [8, NUM_CLASSES] where NUM_CLASSES is the number of classes
