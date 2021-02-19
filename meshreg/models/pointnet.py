#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class PointNet(nn.Module):
    """Shape Encoder using point cloud
        Arguments:
            feature_dim: output feature dimension for the point cloud
        Return:
            A tensor of size NxC, where N is the batch size and C is the feature_dim
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, shapes):
        feats = F.relu(self.bn1(self.conv1(shapes.permute(0, 2, 1))))
        feats = F.relu(self.bn2(self.conv2(feats)))
        feats = self.bn3(self.conv3(feats))
        feats, _ = torch.max(feats, 2)
        feats = feats.view(shapes.size(0), -1)
        return feats
