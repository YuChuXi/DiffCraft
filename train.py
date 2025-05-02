import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LitematicaDataset
from model import DiffCraft
from config import Config


def custom_loss(ns, n):
    # ns: shape (B, X, Y, Z, NS), float tensor from sigmoid
    # n: shape (B, X, Y, Z, N), int tensor with label indices
    print(f"ns shape: {ns.shape}, n shape: {n.shape}")
    B, X, Y, Z, N = n.shape
    NS = ns.size(-1)
    device = ns.device
    
    # 初始化目标张量，全0
    target = torch.zeros_like(ns, device=device)
    print(f"target shape: {target.shape}")
    # 计算全0的mask：检查每个位置的N个标签是否全为0
    mask = (n == 0).all(dim=-1)  # 形状 (B, X, Y, Z)
    print(f"mask shape: {mask.shape}")
    # 处理全0的情况：设置目标第0个通道为1
    target[mask, 0] = 1.0
    print(f"target after setting all-zero mask: {target[mask].shape}")
    
    # 处理非全0的情况：收集所有非0标签并设置对应位置为1
    # 获取所有非0元素的坐标和对应的标签值
    nonzero_coords = torch.nonzero(n)  # 形状 (M, 5)，其中每个元素是 [b, x, y, z, k]
    
    if nonzero_coords.size(0) > 0:
        # 分解坐标
        b, x, y, z, k = nonzero_coords.unbind(dim=1)
        # 获取对应的标签ID
        labels = n[b, x, y, z, k].long()  # 确保标签为整数
        
        # 确保标签在有效范围内（0到NS-1）
        # 注意：假设n中的标签ID都是有效的，即0 <= labels < NS
        # 将对应的目标位置设置为1
        target[b, x, y, z, labels] = 1.0
    
    # 计算二元交叉熵损失
    loss = F.binary_cross_entropy(ns, target)
    
    return loss

def train_step(model, batch, optimizer, device):
    model.train()
    # 准备数据
    x = batch.to(device).long()  # (B,X,Y,Z,1+N_STATE)
    B = x.shape[0]
    
    # 生成随机时间步
    t = torch.randint(0, 1000, (B,), device=device)
    
    # 前向传播
    block_logits, state_logits = model(x, t)
    
    # 方块ID交叉熵损失（需要将目标展平）
    block_loss = F.cross_entropy(
        block_logits.permute(0,4,1,2,3),  # 将类别维度放在第1位 (B,n_blocks,X,Y,Z)
        x[...,0].long()  # 目标需要是Long类型 (B,X,Y,Z)
    )
    
    # 状态标签二元交叉熵
    state_loss = custom_loss(state_logits, x[..., 1:])  # (B,X,Y,Z,n_states)
    total_loss = block_loss + state_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {'block_loss': block_loss.item(), 
            'state_loss': state_loss.item(),
            'total_loss': total_loss.item()}

if __name__ == "__main__":
    # 配置
    config = Config()
    config.use_vae = False  # 初始训练不使用VAE
    
    # 数据集
    dataset = LitematicaDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffCraft(config).to(device)
    # print(model)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练循环
    for epoch in range(100):
        for batch in dataloader:
            loss_dict = train_step(model, batch, optimizer, device)
            print(f"Epoch {epoch} Loss: {loss_dict['total_loss']:.4f}")
