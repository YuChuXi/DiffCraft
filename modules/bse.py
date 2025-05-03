import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockEncoder(nn.Module):
    """
    输入形状: (B, X, Y, Z, 1 + max_n_state)
    输出形状: (B, X, Y, Z, E)
    """
    def __init__(self, n_blocks, n_states, max_n_state, E):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_states = n_states
        self.max_n_state = max_n_state
        self.E = E
        
        # 方块ID的嵌入层
        self.block_embed = nn.Embedding(n_blocks, E)
        # 状态标签的嵌入层（所有状态共享）
        self.state_embed = nn.Embedding(n_states, E)
        
    def forward(self, x):
        # 拆分方块ID和状态标签
        block_ids = x[..., 0].long()  # (B,X,Y,Z)
        state_ids = x[..., 1:].long() # (B,X,Y,Z,max_n_state)
        
        # 方块ID嵌入
        block_emb = self.block_embed(block_ids)  # (B,X,Y,Z,E)
        
        # 状态嵌入处理
        state_weights = self.state_embed(state_ids)  # (B,X,Y,Z,max_n_state,E)
        mask = (state_ids != 0).unsqueeze(-1)       # (B,X,Y,Z,max_n_state,1)
        state_emb = (state_weights * mask).sum(dim=-2)  # (B,X,Y,Z,E)
        
        # 合并嵌入
        total_emb = block_emb + state_emb
        return total_emb

class BlockDecoder(nn.Module):
    """
    输入形状: (B, X, Y, Z, E)
    输出形状: (B, X, Y, Z, 1 + max_n_state)
    """
    def __init__(self, n_blocks, n_states, max_n_state, E, big = False):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_states = n_states
        self.max_n_state = max_n_state
        
        if big:
            # 方块ID预测层
            self.block_decoder = nn.Sequential(
                nn.Linear(E, 512),
                nn.ReLU(),
                nn.Linear(512, n_blocks)
            )
            # 状态标签预测层
            self.state_decoder = nn.Sequential(
                nn.Linear(E, 512),
                nn.ReLU(),
                nn.Linear(512, n_states)
            )
        else:
            self.block_decoder = nn.Linear(E, n_blocks)
            # 状态标签预测层
            self.state_decoder = nn.Linear(E, n_states)
            
    def forward(self, emb):
        # 返回原始logits用于训练
        block_logits = self.block_decoder(emb)  # (B,X,Y,Z,n_blocks)
        state_logits = torch.sigmoid(self.state_decoder(emb))  # (B,X,Y,Z,n_states)
        return block_logits, state_logits
