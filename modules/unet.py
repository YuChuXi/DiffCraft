import torch
from torch import nn

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch//2 + out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DenoiseNet3D(nn.Module):
    def __init__(self, in_dim, text_dim):
        super().__init__()
        self.inc = nn.Conv3d(in_dim, 64, 3, padding=1)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.outc = nn.Conv3d(64, in_dim, 3, padding=1)
        self.text_proj = nn.Linear(text_dim, 256)

    def forward(self, x, t, text_emb):
        # x: (B,X,Y,Z,C)
        x = x.permute(0,4,1,2,3)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x.permute(0,2,3,4,1)