import torch
import torch.nn as nn
import math

class VAEEncoder3D(nn.Module):
    """3D VAE编码器"""
    def __init__(self, in_channels, out_channels, R):
        super().__init__()
        self.downsample_steps = int(math.log2(R))
        layers = []
        current_channels = in_channels
        for _ in range(self.downsample_steps):
            layers += [
                nn.Conv3d(current_channels, current_channels*2, 
                         kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ]
            current_channels *= 2
        layers.append(nn.Conv3d(current_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B,X,Y,Z,C) -> (B,C,X,Y,Z)
        x = x.permute(0,4,1,2,3)
        x = self.net(x)
        return x.permute(0,2,3,4,1)  # (B,X/R,Y/R,Z/R,out_channels)

class VAEDecoder3D(nn.Module):
    """3D VAE解码器"""
    def __init__(self, in_channels, out_channels, R):
        super().__init__()
        self.upsample_steps = int(math.log2(R))
        layers = []
        current_channels = in_channels
        for _ in range(self.upsample_steps):
            layers += [
                nn.ConvTranspose3d(current_channels, current_channels//2,
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            ]
            current_channels = current_channels // 2
        layers.append(nn.Conv3d(current_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B,X,Y,Z,C) -> (B,C,X,Y,Z)
        x = x.permute(0,4,1,2,3)
        x = self.net(x)
        return x.permute(0,2,3,4,1)  # (B,X*R,Y*R,Z*R,out_channels) 