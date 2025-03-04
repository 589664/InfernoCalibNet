import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        # Load pre-trained ResNet18 model
        self.backbone = models.resnet18()

        # Modify conv1 to handle grayscale images
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Create a feature extractor using the backbone's children except the last two layers
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        num_feat = 512  # ResNet18's final feature map has 512 channels

        # New classification layers with dropout
        self.classifier = nn.Sequential(
            nn.Conv2d(num_feat, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),  # Raw logits for BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# Example Usage
# model = ResNetBinaryClassifier()
# print(model)
