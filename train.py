import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LitematicaDataset
from model import DiffCraft
from config import Config
import wandb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time
import math
import Atiny

# plt.ion()
# fig, ax = plt.subplots()
# im = ax.imshow([[0]], aspect='auto', cmap='viridis')
# plt.colorbar(im)
# plt.title('Block Decoder Weights')


def train_step(config: Config, model: DiffCraft, batch, optimizer, device):
    model.train()
    # 准备数据
    voxel = batch["voxel"].to(device).long()
    original_shapes = batch["original_shape"].to(device)
    mask = batch["mask"].to(device)
    B = voxel.shape[0]

    # 生成随机时间步 (扩散步数)
    t = torch.randint(0, config.num_diffusion_steps, (B,), device=device)

    # 前向传播并计算三条路径的联合损失
    loss_dict = model.compute_loss(
        {"voxel": voxel, "original_shape": original_shapes, "mask": mask}, t
    )

    # 反向传播
    optimizer.zero_grad()
    loss_dict["total_loss"].backward()
    optimizer.step()

    return {k: v.item() for k, v in loss_dict.items()}


if __name__ == "__main__":
    # 配置
    config = Config()
    config.use_vae = False  # 初始训练不使用VAE

    # 数据集
    def collate_fn(batch):
        """动态padding并保留原始形状信息"""
        # 收集原始形状
        original_shapes = torch.stack([item["original_shape"] for item in batch])

        # 计算各维度最大值
        max_dims = torch.max(original_shapes, dim=0)[0]
        max_x, max_y, max_z = max_dims.tolist()

        # 初始化padded张量
        padded_batch = []
        masks = []

        for item in batch:
            voxel = item["voxel"]
            if isinstance(voxel, np.ndarray):
                voxel = torch.from_numpy(voxel)
            x, y, z = voxel.shape[:3]

            # 计算padding尺寸
            pad_x = max_x - x
            pad_y = max_y - y
            pad_z = max_z - z

            # 进行padding (只padding空间维度，不padding状态维度)
            padded = F.pad(voxel, (0, 0, 0, pad_z, 0, pad_y, 0, pad_x))
            padded_batch.append(padded)

            # 创建mask (1表示有效区域，0表示padding)
            mask = torch.ones((x, y, z), dtype=torch.bool)
            mask = F.pad(mask, (0, pad_z, 0, pad_y, 0, pad_x), value=False)
            masks.append(mask)

        return {
            "voxel": torch.stack(padded_batch),
            "original_shape": original_shapes,
            "mask": torch.stack(masks),
        }

    dataset = LitematicaDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 可以适当增大batch size
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffCraft(config).to(device)

    print(
        f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M"
    )

    # 优化器
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "atiny":
        optimizer = Atiny.Atiny(
            model.parameters(),
            lr=config.init_lr,
            weight_decay=config.weight_decay,
        )

    wandb.init(
        project="DiffCraft",
        config=config.__dict__,
        name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )
    # 训练循环
    for epoch in tqdm.trange(10000):
        for batch in dataloader:
            if config.lr_schedule == "cos":
                # 余弦退火学习率调度
                lr = (math.cos(math.pi * epoch / 10000) * 0.5 + 0.5) * (
                    config.init_lr - config.final_lr
                ) + config.final_lr
            elif config.lr_schedule == "linear":
                # 线性学习率调度
                lr = (1 - epoch / 10000) * (
                    config.init_lr - config.final_lr
                ) + config.final_lr
            elif config.lr_schedule == "exp":
                # 指数衰减学习率调度
                lr = config.init_lr * (config.final_lr / config.init_lr) ** (
                    epoch / 10000
                )
            else:
                lr = config.init_lr

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss_dict = train_step(config, model, batch, optimizer, device)
            loss_dict.update(
                    {
                        "learning_rate": lr,
                    }
                )
            wandb.log(
                loss_dict
            )

    torch.save(model.state_dict(), "diffcraft.pth")
