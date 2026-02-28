import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for multi-level feature extraction.
    Returns feature maps from 4 stages.
    """

    def __init__(self, name="resnet34", pretrained=True):
        super().__init__()

        if name == "resnet34":
            base = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError("Only resnet34 is supported in this release.")

        self.stage0 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu
        )

        self.stage1 = nn.Sequential(
            base.maxpool,
            base.layer1
        )

        self.stage2 = base.layer2
        self.stage3 = base.layer3
        self.stage4 = base.layer4

        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):

        x = self.stage0(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        return [f1, f2, f3, f4]