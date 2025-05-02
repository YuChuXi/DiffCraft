import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LitematicaDataset
from model import DiffCraft
from config import Config

def train_step(model, batch, optimizer, device):
    model.train()
    # 准备数据
    x = batch.to(device)  # (B,X,Y,Z,1+N_STATE)
    B = x.shape[0]
    
    # 生成随机时间步
    t = torch.randint(0, 1000, (B,), device=device)
    
    # 前向传播
    pred = model(x, t)
    
    # 计算损失（示例：方块ID交叉熵 + 状态标签的二元交叉熵）
    block_loss = F.cross_entropy(pred[...,0], x[...,0].long())
    state_loss = F.binary_cross_entropy_with_logits(
        pred[...,1:], (x[...,1:] !=0).float())
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
    print(model)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练循环
    for epoch in range(100):
        for batch in dataloader:
            loss_dict = train_step(model, batch, optimizer, device)
            print(f"Epoch {epoch} Loss: {loss_dict['total_loss']:.4f}")