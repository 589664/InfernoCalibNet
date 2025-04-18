import torch.nn as nn

from torchvision.models import resnet34, resnet50
from torchvision.models import ResNet34_Weights, ResNet50_Weights


#=======================================================================================================================
# 🚀 Main Code
#=======================================================================================================================

class InfernoCalibNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.6, model_type='resnet34', pretrained=True):
        super(InfernoCalibNet, self).__init__()

        if model_type == 'resnet34':
            if pretrained:
                base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                base = resnet34(weights=None)
        elif model_type == 'resnet50':
            if pretrained:
                base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                base = resnet50(weights=None)
        else:
            raise ValueError('model_type must be "resnet34" or "resnet50"')

        base.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.base_model = nn.Sequential(*list(base.children())[:-2])

        num_feat = base.fc.in_features

        self.classifier = nn.Sequential(
            nn.Conv2d(num_feat, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x