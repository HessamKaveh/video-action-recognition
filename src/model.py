import torch
import torch.nn as nn


class Conv3dBlock(nn.Module):
    """3D convolution block"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ActionRecognitionNet(nn.Module):
    """3D CNN for video action recognition"""
    def __init__(self, num_classes=5, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        
        # 3D Convolutional layers
        self.conv1 = Conv3dBlock(3, 32, kernel_size=(3, 3, 3))  # (T, H, W)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv2 = Conv3dBlock(32, 64, kernel_size=(3, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3 = Conv3dBlock(64, 128, kernel_size=(3, 3, 3))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv4 = Conv3dBlock(128, 256, kernel_size=(3, 3, 3))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1))  # Keep temporal dimension
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (B, 3, T, H, W) - Batch, Channels, Time, Height, Width
        
        # Conv block 1
        x = self.conv1(x)  # (B, 32, T, H, W)
        x = self.pool1(x)  # (B, 32, T/2, H/2, W/2)
        
        # Conv block 2
        x = self.conv2(x)  # (B, 64, T/2, H/2, W/2)
        x = self.pool2(x)  # (B, 64, T/4, H/4, W/4)
        
        # Conv block 3
        x = self.conv3(x)  # (B, 128, T/4, H/4, W/4)
        x = self.pool3(x)  # (B, 128, T/8, H/8, W/8)
        
        # Conv block 4
        x = self.conv4(x)  # (B, 256, T/8, H/8, W/8)
        x = self.pool4(x)  # (B, 256, T/8, H/8, W/8)
        
        # Global pooling
        x = self.global_pool(x)  # (B, 256, 1, 1, 1)
        x = x.flatten(1)  # (B, 256)
        
        # Classification
        x = self.fc1(x)  # (B, 128)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, num_classes)
        
        return x
