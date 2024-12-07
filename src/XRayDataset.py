import torch
import torch.nn as nn
from torchvision.models import (
    densenet201,
    efficientnet_b4,
    EfficientNet_B4_Weights,
    DenseNet201_Weights,
)
from config import NUM_CL, DROP_PROB


class DenseNet201(nn.Module):
    def __init__(self) -> None:
        super(DenseNet201, self).__init__()

        # Load pre-trained DenseNet201 model
        self.densenet = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)

        # Extract the input features of the classifier layer
        in_features = self.densenet.classifier.in_features

        # Replace the classifier layer with a custom one for the desired number of classes, adding Dropout
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(DROP_PROB),  # Dropout with configurable probability
            nn.Linear(in_features, NUM_CL),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)


# Example usage for DenseNet201:
# model = DenseNet201()
# input_tensor = torch.randn(8, 3, 224, 224)  # Batch of 8 RGB images of size 224x224
# output = model(input_tensor)
# print(output.shape)  # Should be [8, NUM_CL] where NUM_CL is the number of classes


class EfficientNetB4(nn.Module):
    def __init__(self) -> None:
        super(EfficientNetB4, self).__init__()

        # Load pre-trained EfficientNet B4 model
        self.effnet = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

        # Extract the input features of the classifier layer
        in_features = self.effnet.classifier[1].in_features

        # Replace the classifier layer with a custom one for the desired number of classes, adding Dropout
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(DROP_PROB),  # Dropout with configurable probability
            nn.Linear(in_features, NUM_CL),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.effnet(x)


# Example usage for EfficientNetB4:
# model = EfficientNetB4()
# input_tensor = torch.randn(8, 3, 380, 380)  # Batch of 8 RGB images of size 380x380
# output = model(input_tensor)
# print(output.shape)  # Should be [8, NUM_CL] where NUM_CL is the number of classes
