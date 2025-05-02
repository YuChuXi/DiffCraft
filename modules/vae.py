from torch import nn


def adaptive_pad(x, target_ratio):
    """动态填充至能被ratio整除"""
    B, C, D, H, W = x.shape
    pad_d = (target_ratio - D % target_ratio) % target_ratio
    pad_h = (target_ratio - H % target_ratio) % target_ratio
    pad_w = (target_ratio - W % target_ratio) % target_ratio
    return nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))


class VAEEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=512, ratios=2):
        super().__init__()
        self.ratios = ratios
        self.encoders = nn.ModuleDict(
            {
                f"x{ratio}": nn.Sequential(
                    nn.Conv3d(in_dim, latent_dim // 4, 3, stride=2, padding=1),
                    nn.GroupNorm(8, latent_dim // 4),
                    nn.GELU(),
                    nn.Conv3d(latent_dim // 4, latent_dim, 3, stride=2, padding=1),
                )
                for ratio in ratios
            }
        )

    def forward(self, x, ratio):
        x = adaptive_pad(x, ratio)
        return self.encoders[f"x{ratio}"](x)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=512, out_dim=256, ratios=[2, 3, 4]):
        super().__init__()
        self.decoders = nn.ModuleDict(
            {
                f"x{ratio}": nn.Sequential(
                    nn.ConvTranspose3d(
                        latent_dim, latent_dim // 4, 3, stride=2, padding=1
                    ),
                    nn.GroupNorm(8, latent_dim // 4),
                    nn.GELU(),
                    nn.ConvTranspose3d(
                        latent_dim // 4, out_dim, 3, stride=2, padding=1
                    ),
                )
                for ratio in ratios
            }
        )

    def forward(self, z, ratio):
        return self.decoders[f"x{ratio}"](z)
