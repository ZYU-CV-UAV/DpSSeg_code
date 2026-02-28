import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """
    Hypercolumn-style multi-scale feature fusion.
    All features are projected and upsampled to the same resolution.
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in in_channels_list
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels * len(in_channels_list),
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):

        target_size = features[0].shape[2:]
        resized_features = []

        for feat, proj in zip(features, self.projections):
            x = proj(feat)
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )
            resized_features.append(x)

        hypercolumn = torch.cat(resized_features, dim=1)

        return self.fuse(hypercolumn)