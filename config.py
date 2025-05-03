import torch

class Config:
    def __init__(self):
        # BlockEncoder/Decoder 参数
        self.n_blocks = 1535       # 方块ID的种类数
        self.n_states = 511       # 每个方块状态标签的种类数
        self.max_n_state = 7          # 每个方块的最大状态标签数
        self.E = 128               # 嵌入维度
        
        # VAE 参数
        self.use_vae = False      # 是否使用VAE
        self.C = 32               # 潜在空间维度
        self.R = 8                # 下采样因子 (必须是2的幂)
        
        # DenoiseNet3D 参数
        self.model_channels = 64  # 模型通道数
        self.num_res_blocks = 2
        self.attention_resolutions = [4]  # 应用注意力的分辨率
        self.channel_mult = (1, 2, 2, 4)        # 各阶段通道倍增系数
        self.num_heads = 4                   # 注意力头数
        self.text_emb_dim = 768               # 文本嵌入维度