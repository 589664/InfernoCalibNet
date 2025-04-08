import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class InfernoCalibNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.6):
        super(InfernoCalibNet, self).__init__()

        # base_model = resnet34()
        base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        base_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.base_model = nn.Sequential(*list(base_model.children())[:-2])

        num_feat = base_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Conv2d(num_feat, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(128, num_classes),
        )

        # self.classifier = nn.Sequential(
        #     nn.Conv2d(num_feat, 512, kernel_size=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Dropout(drop_rate),
        #     nn.Conv2d(512, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Dropout(drop_rate),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(128, num_classes),
        # )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x
