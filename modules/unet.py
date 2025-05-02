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

    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim, echo=""):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.echo = echo
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.text_mlp = nn.Linear(text_emb_dim, out_channels)
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
        )
        self.res_conv = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb, text_emb):
        print(
            self.echo,
            self.in_channels,
            self.out_channels,
            x.shape,
            t_emb.shape,
            text_emb.shape,
        )
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
    """3D去噪网络（修正维度版本）"""

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
        self.text_proj = (
            nn.Linear(config.text_emb_dim, config.model_channels)
            if config.text_emb_dim
            else None
        )

        # 输入层 (B, C, X, Y, Z)
        self.input_conv = nn.Conv3d(
            config.E, config.model_channels, kernel_size=3, padding=1
        )

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        ch_mult = config.channel_mult
        current_res = 1
        in_ch = config.model_channels

        # 预计算各阶段通道数
        channels = [config.model_channels]
        for i, mult in enumerate(ch_mult):
            out_ch = mult * config.model_channels
            for _ in range(config.num_res_blocks):
                channels.append(out_ch)
            if i != len(ch_mult) - 1:
                channels.append(out_ch)  # 下采样层
                current_res *= 2

        # 构建下采样
        idx = 0
        for i, mult in enumerate(ch_mult):
            out_ch = mult * config.model_channels
            for _ in range(config.num_res_blocks):
                layers = [
                    ResidualBlock3D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        text_emb_dim=config.model_channels,
                        time_emb_dim=config.model_channels,
                        echo=f"D{i}_res{_}",
                    )
                ]
                if current_res in config.attention_resolutions:
                    layers.append(AttentionBlock3D(out_ch))
                self.down_blocks.append(nn.Sequential(*layers))
                in_ch = out_ch
                idx += 1

            # 下采样层（非最后阶段）
            if i != len(ch_mult) - 1:
                self.down_blocks.append(
                    nn.Sequential(
                        nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
                    )
                )
                idx += 1

        # 中间块
        self.mid_blocks = nn.ModuleList(
            [
                ResidualBlock3D(
                    in_ch, in_ch, config.model_channels, config.model_channels, "mid1"
                ),
                AttentionBlock3D(in_ch),
                ResidualBlock3D(
                    in_ch, in_ch, config.model_channels, config.model_channels, "mid2"
                ),
            ]
        )

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = mult * config.model_channels
            for _ in range(config.num_res_blocks + 1):  # +1用于跳跃连接
                is_attention = current_res in config.attention_resolutions
                layers = [
                    ResidualBlock3D(
                        in_channels=in_ch + channels.pop(),  # 跳跃连接
                        out_channels=out_ch,
                        text_emb_dim=config.model_channels,
                        time_emb_dim=config.model_channels,
                        echo=f"U{i}_res{_}",
                    )
                ]
                if is_attention:
                    layers.append(AttentionBlock3D(out_ch))
                self.up_blocks.append(nn.Sequential(*layers))
                in_ch = out_ch

            # 上采样层（非最后阶段）
            if i != 0:
                self.up_blocks.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(
                            in_ch,
                            in_ch,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        )
                    )
                )

        # 输出层
        self.out_conv = nn.Conv3d(
            config.model_channels, config.E, kernel_size=3, padding=1
        )

    def forward(self, x, timesteps, text_emb=None):
        B, X, Y, Z, E = x.shape

        # 时间条件
        t_emb = self.time_embed(timesteps)  # (B, model_channels)

        # 文本条件
        c_emb = (
            self.text_proj(text_emb)
            if text_emb
            else torch.zeros(x.shape[0], self.config.model_channels, device=x.device)
        )  # (B, model_channels)

        # 输入转换 (B, E, X, Y, Z)
        h = x.permute(0, 4, 1, 2, 3)
        h = self.input_conv(h)  # (B, model_channels, X, Y, Z)

        # 存储跳跃连接
        skips = [h]

        # 下采样
        for block in self.down_blocks:
            h = (
                block[0](h, t_emb, c_emb)
                if isinstance(block[0], ResidualBlock3D)
                else block(h)
            )
            skips.append(h)

        # 中间块
        for block in self.mid_blocks:
            h = (
                block(h, t_emb, c_emb)
                if isinstance(block, ResidualBlock3D)
                else block(h)
            )

        # 上采样
        for block in self.up_blocks:
            if isinstance(block[0], ResidualBlock3D):
                skip = skips.pop()
                # 对齐空间维度
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb, c_emb)
            else:
                h = block(h)

        # 输出 (B, E, X, Y, Z) -> (B, X, Y, Z, E)
        return self.out_conv(h).permute(0, 2, 3, 4, 1)
