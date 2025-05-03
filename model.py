import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.bse import BlockEncoder, BlockDecoder
from modules.vae import VAEEncoder3D, VAEDecoder3D
from modules.unet import DenoiseNet3D


def block_state_loss(ns, n):
    # ns: shape (B, X, Y, Z, NS), float tensor from sigmoid
    # n: shape (B, X, Y, Z, N), int tensor with label indices
    assert (
        ns.shape[:-1] == n.shape[:-1]
    ), f"ns shape {ns.shape} and n shape {n.shape} do not match"
    device = ns.device

    # 初始化目标张量，全0
    target = torch.zeros_like(ns, device=device)
    # 计算全0的mask：检查每个位置的N个标签是否全为0
    mask = (n == 0).all(dim=-1)  # 形状 (B, X, Y, Z)
    target[mask][0] = 1.0  # 将全0的mask位置设置为1

    # 处理非全0的情况：收集所有非0标签并设置对应位置为1
    # 获取所有非0元素的坐标和对应的标签值
    nonzero_coords = torch.nonzero(n)  # 形状 (M, 5)，其中每个元素是 [b, x, y, z, k]

    if nonzero_coords.size(0) > 0:
        # 分解坐标
        pos = nonzero_coords.unbind(dim=1)
        # 获取对应的标签ID
        labels = n[pos].long()  # 确保标签为整数

        # 确保标签在有效范围内（0到NS-1）
        # 注意：假设n中的标签ID都是有效的，即0 <= labels < NS
        # 将对应的目标位置设置为1
        target[*pos[:-1], labels] = 1.0

    # 计算二元交叉熵损失
    loss = F.binary_cross_entropy(ns, target)
    # 计算准确率：预测值经过阈值处理并与目标比较
    pred = (ns >= 0.5)  # 使用0.5作为阈值
    accuracy = (pred == target).float().mean()  # 计算正确率

    return loss, accuracy  # 返回损失和准确率

class DiffCraft(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化各个模块
        self.block_encoder = BlockEncoder(
            config.n_blocks, config.n_states, config.embed_dim
        )
        self.block_decoder = BlockDecoder(
            config.n_blocks, config.n_states, config.embed_dim
        )

        # VAE模块
        self.use_vae = config.use_vae
        if self.use_vae:
            self.vae_encoder = VAEEncoder3D(
                config.embed_dim, config.latent_dim, config.downsample_ratio
            )
            self.vae_decoder = VAEDecoder3D(
                config.latent_dim, config.embed_dim, config.downsample_ratio
            )

        # 去噪网络
        self.denoise_net = DenoiseNet3D(config)

        # 扩散训练相关参数
        self.register_buffer(
            "betas",
            torch.linspace(
                config.beta_start, config.beta_end, config.num_diffusion_steps
            ),
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def pad_to_multiple(self, x, R):
        """将输入填充到R的倍数"""
        pad = []
        for d in x.shape[1:4]:  # X,Y,Z维度
            remainder = d % R
            pad.append(0 if remainder == 0 else R - remainder)
        x_padded = F.pad(x, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))
        return x_padded, pad

    def unpad(self, x, pad):
        """去除填充部分"""
        x_unpadded = x[:, : -pad[0], : -pad[1], : -pad[2], :] if any(pad) else x
        return x_unpadded

    def add_noise(self, x_start, t):
        """前向加噪过程"""
        # 获取系数的累积乘积
        sqrt_alpha_cumprod = (self.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = ((1 - self.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1, 1)
        
        # 生成噪声
        noise = torch.randn_like(x_start)
        
        # 添加噪声
        x_noisy = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        return x_noisy, noise

    def compute_loss(self, x, t, text_emb=None, skip_unet=False, keep_bse_vae=False):
        """计算三条路径的联合损失"""
        # 路径1: 体素重建路径
        # 从输入字典中获取各字段
        block_ids = x["voxel"][..., 0].long()  # 方块ID
        state_ids = x["voxel"][..., 1:].long()  # 状态标签
        original_shapes = x["original_shape"]
        mask = x["mask"]

        # 编码体素数据
        emb = self.block_encoder(block_ids, state_ids, mask)
        block_logits, state_logits = self.block_decoder(emb, mask)
        # print(
        #     block_ids.shape,
        #     state_ids.shape,
        #     emb.shape,
        #     block_logits.shape,
        #     state_logits.shape,
        # ) # torch.Size([1, 325, 1, 329]) torch.Size([1, 325, 1, 329, 7]) torch.Size([1, 325, 1, 329, 64]) torch.Size([1, 325, 1, 329, 1535]) torch.Size([1, 325, 1, 329, 511])
            
        block_logits_flat = block_logits.view(-1, block_logits.shape[-1])  # (B*X*Y*Z, C)
        block_ids_flat = block_ids.long().view(-1)                        # (B*X*Y*Z)
        mask_flat = mask.view(-1)                                         # (B*X*Y*Z)

        # 计算交叉熵损失
        block_loss = F.cross_entropy(
            block_logits_flat,
            block_ids_flat,
            reduction='none'
        )

        # 应用mask并计算加权损失
        block_loss = (block_loss * mask_flat).sum() / mask.sum()
        
        # 计算mask区域的准确率
        preds = block_logits.argmax(dim=-1)
        block_accuracy = (preds[mask] == block_ids[mask]).float().mean()
        
        # 应用mask处理state loss
        state_loss, state_accuracy = block_state_loss(state_logits[mask], x["voxel"][..., 1:][mask])


        # 路径2: VAE重建路径
        if self.use_vae:
            # VAE前向计算并获取KL散度
            latent = self.vae_encoder(emb)
            mu, log_var = None, None # TODO
            pred_emb = self.vae_decoder(latent)

            # 带mask的重建损失
            mask_expanded = mask.unsqueeze(-1)  # (B,X,Y,Z,1)
            recon_loss = F.mse_loss(pred_emb[mask_expanded], emb[mask_expanded])

            # 计算KL散度（按有效体素数量归一化）
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss = kl_loss / (mask.sum() + 1e-8)  # 防止除零

            # 带温度系数的beta调整
            log_var = torch.clamp(log_var, min=-20, max=20)  # 防止数值溢出
            valid_ratio = mask.sum() / mask.numel()
            adaptive_beta = self.config.vae_beta * torch.sigmoid(10 * valid_ratio)

            # 组合损失项并添加稳定性检查
            vae_loss = recon_loss * 0.5 + adaptive_beta * kl_loss
            assert not torch.isnan(vae_loss), "VAE loss出现NaN"

            # 保留中间结果用于监控
            vae_stats = {
                "vae_recon": recon_loss.item(),
                "vae_kl": kl_loss.item(),
                "vae_beta": adaptive_beta.item(),
            }
        else:
            vae_loss = torch.tensor(0.0, device=x["voxel"].device)
            vae_stats = {}

        # 路径3: 扩散去噪路径
        if not skip_unet:
            if self.use_vae:
                clean_latent = latent
            else:
                clean_latent = emb
            if keep_bse_vae:
                clean_latent.detach_()  # 保持BSE和VAE的计算图不变

            noisy_latent, noise = self.add_noise(clean_latent, t)
            pred_noise = self.denoise_net(
                noisy_latent, t, text_emb=text_emb, original_shapes=original_shapes
            )
            denoise_loss = F.mse_loss(pred_noise, noise)
        else:
            denoise_loss = torch.tensor(0.0, device=x["voxel"].device)

        loss_dict = {
            "total_loss": block_loss + state_loss + vae_loss + denoise_loss,
            "block_loss": block_loss,
            "state_loss": state_loss,
            "block_accuracy": block_accuracy,
            "state_accuracy": state_accuracy,
            "vae_loss": vae_loss,
            "denoise_loss": denoise_loss,
        }
        loss_dict.update(vae_stats)
        return loss_dict

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
            latent = torch.randn(
                shape[0], self.config.latent_dim, *shape[2:], device=self.betas.device
            )
        else:
            latent = torch.randn(shape, device=self.betas.device)

        for t in reversed(range(timesteps)):
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            sqrt_one_minus_alpha_cumprod_t = (1 - self.alphas_cumprod[t]) ** 0.5

            # 预测噪声
            with torch.no_grad():
                pred_noise = self.denoise_net(
                    latent,
                    torch.full((latent.size(0),), t, device=latent.device),
                    text_emb,
                )

            # 更新潜在表示
            latent = (latent - beta_t * pred_noise / sqrt_one_minus_alpha_cumprod_t) / (
                alpha_t**0.5
            )
            if t > 0:
                latent += torch.randn_like(latent) * beta_t**0.5

        # 最终解码
        if self.use_vae:
            return self.block_decoder(self.vae_decoder(latent))
        return self.block_decoder(latent)
