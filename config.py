import torch

class Config:
    def __init__(self):
        # Block编码/解码参数
        self.n_blocks = 1535          # 方块ID种类数
        self.n_states = 511           # 方块状态标签种类数
        self.max_n_state = 7          # 方块最大状态标签数
        self.embed_dim = 64           # 嵌入维度
        
        # VAE参数
        self.use_vae = False          # 是否使用VAE
        self.latent_dim = 32          # 潜在空间维度
        self.downsample_ratio = 8     # 下采样因子
        
        # 扩散模型参数
        self.num_diffusion_steps = 1000  # 扩散步数
        self.beta_start = 0.0001      # beta起始值
        self.beta_end = 0.02          # beta结束值
        
        # DenoiseNet3D 参数
        self.model_channels = 64  # 模型通道数
        self.num_res_blocks = 2
        self.attention_resolutions = ()  # 应用注意力的分辨率
        self.channel_mult = (1, 2, 4, 8)        # 各阶段通道倍增系数
        self.num_heads = 4                   # 注意力头数
        self.text_emb_dim = 768               # 文本嵌入维度
        
        # 训练参数
        self.optimizer = "adamw"  # 优化器类型
        self.init_lr = 1e-4        # 初始学习率
        self.final_lr = 1e-5        # 最终学习率
        self.lr_schedule = "cos"  # 学习率调度器类型
        self.weight_decay = 1e-5  # 权重衰减
        self.skip_unet = False
        self.keep_bse_vae = False
        self.batch_size = 1  # 批大小(不建议大于1, 不同输入的padding会浪费显存)
        self.grad_accumulation_steps = 16  # 梯度累积步数
