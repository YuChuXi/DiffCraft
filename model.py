import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.bse import BlockEncoder, BlockDecoder
from modules.vae import VAEEncoder3D, VAEDecoder3D
from modules.unet import DenoiseNet3D

class DiffCraft(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化各个模块
        self.block_encoder = BlockEncoder(
            config.n_blocks, config.n_states, config.max_n_state, config.embed_dim)
        self.block_decoder = BlockDecoder(
            config.n_blocks, config.n_states, config.max_n_state, config.embed_dim)
        
        # VAE模块
        self.use_vae = config.use_vae
        if self.use_vae:
            self.vae_encoder = VAEEncoder3D(config.embed_dim, config.latent_dim, config.downsample_ratio)
            self.vae_decoder = VAEDecoder3D(config.latent_dim, config.embed_dim, config.downsample_ratio)
        
        # 去噪网络
        self.denoise_net = DenoiseNet3D(config)
        
        # 扩散训练相关参数
        self.register_buffer('betas', torch.linspace(config.beta_start, config.beta_end, config.num_diffusion_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def pad_to_multiple(self, x, R):
        """将输入填充到R的倍数"""
        pad = []
        for d in x.shape[1:4]:  # X,Y,Z维度
            remainder = d % R
            pad.append(0 if remainder ==0 else R - remainder)
        x_padded = F.pad(x, (0,0, 0,pad[2], 0,pad[1], 0,pad[0]))
        return x_padded, pad
    
    def unpad(self, x, pad):
        """去除填充部分"""
        x_unpadded = x[:, :-pad[0], :-pad[1], :-pad[2], :] if any(pad) else x
        return x_unpadded
    
    def add_noise(self, x_start, t):
        """前向加噪过程"""
        sqrt_alpha_cumprod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]) ** 0.5
        
        noise = torch.randn_like(x_start)
        x_noisy = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        return x_noisy, noise

    def compute_loss(self, x, t, text_emb=None):
        """计算三条路径的联合损失"""
        # 路径1: 体素重建路径
        emb = self.block_encoder(x)
        x_pred = self.block_decoder(emb)
        recon_loss = F.mse_loss(x_pred, x)

        # 路径2: VAE重建路径
        if self.use_vae:
            latent = self.vae_encoder(emb)
            pred_emb = self.vae_decoder(latent)
            vae_loss = F.mse_loss(pred_emb, emb)
        else:
            vae_loss = torch.tensor(0.0, device=x.device)

        # 路径3: 扩散去噪路径
        if self.use_vae:
            clean_latent = self.vae_encoder(emb.detach())
        else:
            clean_latent = emb.detach()
            
        noisy_latent, noise = self.add_noise(clean_latent, t)
        pred_noise = self.denoise_net(noisy_latent, t, text_emb)
        noise_loss = F.mse_loss(pred_noise, noise)

        return {
            "total": recon_loss + vae_loss + noise_loss,
            "recon": recon_loss,
            "vae": vae_loss,
            "noise": noise_loss
        }

    def forward(self, x, timesteps, text_emb=None, training_mode=True):
        """支持多阶段扩散过程的前向传播"""
        if training_mode:
            return self.compute_loss(x, timesteps, text_emb)
            
        # 推理时使用多步采样
        return self.sample(x.shape, timesteps, text_emb)

    def sample(self, shape, timesteps, text_emb=None):
        """扩散采样过程"""
        # 初始化随机噪声
        if self.use_vae:
            latent = torch.randn(shape[0], self.config.latent_dim, *shape[2:], device=self.betas.device)
        else:
            latent = torch.randn(shape, device=self.betas.device)

        for t in reversed(range(timesteps)):
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            sqrt_one_minus_alpha_cumprod_t = (1 - self.alphas_cumprod[t]) ** 0.5

            # 预测噪声
            with torch.no_grad():
                pred_noise = self.denoise_net(latent, 
                    torch.full((latent.size(0),), t, device=latent.device), 
                    text_emb)

            # 更新潜在表示
            latent = (latent - beta_t * pred_noise / sqrt_one_minus_alpha_cumprod_t) / (alpha_t ** 0.5)
            if t > 0:
                latent += torch.randn_like(latent) * beta_t ** 0.5

        # 最终解码
        if self.use_vae:
            return self.block_decoder(self.vae_decoder(latent))
        return self.block_decoder(latent)
