import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet import ResNetBackbone
from models.fusion.multiscale_fusion import MultiScaleFusion
from models.heads.segmentation_head import SegmentationHead


class DpSSeg(nn.Module):
    """
    Deeply Supervised Multi-Scale Fusion Network (DpSSeg).
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = ResNetBackbone(
            name="resnet34",
            pretrained=True
        )

        self.fusion = MultiScaleFusion(
            in_channels_list=self.backbone.out_channels,
            out_channels=256
        )

        self.main_head = SegmentationHead(
            in_channels=256,
            num_classes=num_classes
        )

        # Deep supervision heads
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(128, num_classes, 1),
            nn.Conv2d(256, num_classes, 1),
            nn.Conv2d(512, num_classes, 1)
        ])

    def forward(self, x):

        features = self.backbone(x)
        f1, f2, f3, f4 = features

        fused = self.fusion(features)

        main_out = self.main_head(fused)

        aux_outs = []

        for feat, head in zip([f2, f3, f4], self.aux_heads):
            aux = head(feat)
            aux = F.interpolate(
                aux,
                size=main_out.shape[2:],
                mode="bilinear",
                align_corners=False
            )
            aux_outs.append(aux)

        return main_out, aux_outs