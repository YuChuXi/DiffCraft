import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock3D(nn.Module):
    """带有条件嵌入的3D残差块"""

    def __init__(self, in_c, out_c, time_emb_dim, text_emb_dim, echo=""):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.echo = echo
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.text_mlp = nn.Linear(text_emb_dim, out_c)
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_c),
            nn.SiLU(),
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(32, out_c),
            nn.SiLU(),
            nn.Conv3d(out_c, out_c, 3, padding=1),
        )
        self.res_conv = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb, text_emb):
        print(self.echo, self.in_c, self.out_c, x.shape, t_emb.shape, text_emb.shape)
        h = self.block(x)
        # 时间条件
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 文本条件
        text_emb = self.text_mlp(text_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.res_conv(x) + h + t_emb + text_emb


class AttentionBlock3D(nn.Module):
    """3D多头注意力块"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, -1).permute(2, 0, 1)  # [N, B, C]
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(1, 2, 0).view(B, C, D, H, W)
        return x + self.proj(attn_out)


class DenoiseNet3D(nn.Module):
    """3D去噪网络"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(config.model_channels),
            nn.Linear(config.model_channels, 4 * config.model_channels),
            nn.SiLU(),
            nn.Linear(4 * config.model_channels, config.model_channels),
        )

        # 文本嵌入
        self.text_embed = nn.Linear(config.text_emb_dim, config.model_channels)

        # 输入层
        self.input_conv = nn.Conv3d(config.E, config.model_channels, 3, padding=1)

        # 下采样
        self.down_blocks = nn.ModuleList()
        current_res = 1
        current_ch = config.model_channels
        ch_mult = config.channel_mult
        for i, mult in enumerate(ch_mult):
            for _ in range(config.num_res_blocks):
                layers = [
                    ResidualBlock3D(
                        current_ch,
                        mult * config.model_channels,
                        config.model_channels,
                        config.model_channels,
                        f"D{i}x{mult}x{_}",
                    )
                ]
                if current_res in config.attention_resolutions:
                    layers.append(AttentionBlock3D(mult * config.model_channels))
                self.down_blocks.append(nn.ModuleList(layers))
                current_ch = mult * config.model_channels
            if i != len(ch_mult) - 1:
                self.down_blocks.append(
                    nn.ModuleList(
                        [nn.Conv3d(current_ch, current_ch, 3, stride=2, padding=1)]
                    )
                )
                current_res *= 2

        # 中间块
        self.mid_blocks = nn.ModuleList(
            [
                ResidualBlock3D(
                    current_ch,
                    current_ch,
                    config.model_channels,
                    config.model_channels,
                    "M1",
                ),
                AttentionBlock3D(current_ch),
                ResidualBlock3D(
                    current_ch,
                    current_ch,
                    config.model_channels,
                    config.model_channels,
                    "M2",
                ),
            ]
        )

        # 上采样
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            for j in range(config.num_res_blocks + 1):
                layers = [
                    ResidualBlock3D(
                        current_ch + (mult * config.model_channels if j == 0 else 0),
                        mult * config.model_channels,
                        config.model_channels,
                        config.model_channels,
                        f"U{i}x{mult}x{j}",
                    )
                ]
                if current_res in config.attention_resolutions:
                    layers.append(AttentionBlock3D(mult * config.model_channels))
                self.up_blocks.append(nn.ModuleList(layers))
                current_ch = mult * config.model_channels
            if i != 0:
                self.up_blocks.append(
                    nn.ModuleList(
                        [
                            nn.ConvTranspose3d(
                                current_ch,
                                current_ch,
                                3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                            )
                        ]
                    )
                )
                current_res //= 2

        # 输出层
        self.out_conv = nn.Conv3d(config.model_channels, config.E, 3, padding=1)

    def forward(self, x, timesteps, text_emb=None):
        # 时间条件
        t_emb = self.time_embed(timesteps)
        # 文本条件
        c_emb = (
            self.text_embed(text_emb)
            if text_emb is not None
            else torch.zeros(x.shape[0], self.config.model_channels, device=x.device)
        )

        # 输入转换
        h = x.permute(0, 4, 1, 2, 3)  # (B,C,X,Y,Z)
        h = self.input_conv(h)

        # 存储中间结果
        hs = [h]

        # 下采样
        for layers in self.down_blocks:
            for layer in layers:
                if isinstance(layer, ResidualBlock3D):
                    h = layer(h, t_emb, c_emb)
                elif isinstance(layer, AttentionBlock3D):
                    h = layer(h)
                else:
                    h = layer(h)
            hs.append(h)

        # 中间块
        for layer in self.mid_blocks:
            if isinstance(layer, ResidualBlock3D):
                h = layer(h, t_emb, c_emb)
            else:
                h = layer(h)

        # 上采样
        for layers in self.up_blocks:
            for layer in layers:
                if isinstance(layer, ResidualBlock3D):
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = layer(h, t_emb, c_emb)
                elif isinstance(layer, AttentionBlock3D):
                    h = layer(h)
                else:
                    h = layer(h)

        # 输出
        h = self.out_conv(h).permute(0, 2, 3, 4, 1)
        return h
