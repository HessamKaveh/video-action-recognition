"""Spatio-temporal CNNs from torchvision (Tran et al., CVPR 2018), Kinetics-400 pretrained or from scratch.

  r3d_18       : full 3D convolutions (3x3x3)
  r2plus1d_18  : (2+1)D factorisation: 1x3x3 spatial conv followed by 3x1x1 temporal conv
"""
import torch.nn as nn
from torchvision.models import video as tvv

ARCHS = {
    "r3d_18": (tvv.r3d_18, tvv.R3D_18_Weights.KINETICS400_V1),
    "r2plus1d_18": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights.KINETICS400_V1),
}


def build(arch="r2plus1d_18", num_classes=101, pretrained=True, dropout=0.5):
    fn, w = ARCHS[arch]
    m = fn(weights=w if pretrained else None)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    return m
