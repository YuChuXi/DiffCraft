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
            config.n_blocks, config.n_states, config.max_n_state, config.E)
        self.block_decoder = BlockDecoder(
            config.n_blocks, config.n_states, config.max_n_state, config.E)
        
        # VAE模块
        self.use_vae = config.use_vae
        if self.use_vae:
            self.vae_encoder = VAEEncoder3D(config.E, config.C, config.R)
            self.vae_decoder = VAEDecoder3D(config.C, config.E, config.R)
        
        # 去噪网络
        self.denoise_net = DenoiseNet3D(config)
        
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
    
    def forward(self, x, timesteps, text_emb=None, skip_unet=False):
        # 1. 编码
        emb = self.block_encoder(x)  # (B,X,Y,Z,E)
        
        if skip_unet:
            return self.block_decoder(emb)
        
        # 2. VAE编码（可选）
        if self.use_vae:
            # 填充
            emb_padded, pad = self.pad_to_multiple(emb, self.config.R)
            # VAE编码
            latent = self.vae_encoder(emb_padded)
            # 去噪
            latent_denoised = self.denoise_net(latent, timesteps, text_emb)
            # VAE解码
            emb_decoded_padded = self.vae_decoder(latent_denoised)
            # 去除填充
            emb_decoded = self.unpad(emb_decoded_padded, pad)
        else:
            # 直接去噪
            emb_denoised = self.denoise_net(emb, timesteps, text_emb)
            emb_decoded = emb_denoised
        
        # 3. 解码
        output = self.block_decoder(emb_decoded)
        return output