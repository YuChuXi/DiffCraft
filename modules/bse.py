import torch
import torch.nn as nn

class BlockEncoder(nn.Module):
    """
    输入形状: (B, X, Y, Z, 1 + N_STATE)
    输出形状: (B, X, Y, Z, E)
    """
    def __init__(self, n_blocks, n_states, N_STATE, E):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_states = n_states
        self.N_STATE = N_STATE
        self.E = E
        
        # 方块ID的嵌入层
        self.block_embed = nn.Embedding(n_blocks, E)
        # 状态标签的嵌入层（所有状态共享）
        self.state_embed = nn.Embedding(n_states, E)
        
    def forward(self, x):
        # 拆分方块ID和状态标签
        block_ids = x[..., 0].long()  # (B,X,Y,Z)
        state_ids = x[..., 1:].long() # (B,X,Y,Z,N_STATE)
        
        # 方块ID嵌入
        block_emb = self.block_embed(block_ids)  # (B,X,Y,Z,E)
        
        # 状态嵌入处理
        state_weights = self.state_embed(state_ids)  # (B,X,Y,Z,N_STATE,E)
        mask = (state_ids != 0).unsqueeze(-1)       # (B,X,Y,Z,N_STATE,1)
        state_emb = (state_weights * mask).sum(dim=-2)  # (B,X,Y,Z,E)
        
        # 合并嵌入
        total_emb = block_emb + state_emb
        return total_emb

class BlockDecoder(nn.Module):
    """
    输入形状: (B, X, Y, Z, E)
    输出形状: (B, X, Y, Z, 1 + N_STATE)
    """
    def __init__(self, n_blocks, n_states, N_STATE, E):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_states = n_states
        self.N_STATE = N_STATE
        
        # 方块ID预测层
        self.block_decoder = nn.Linear(E, n_blocks)
        # 状态标签预测层
        self.state_decoder = nn.Linear(E, n_states)
        
    def forward(self, emb):
        # 预测方块ID
        block_logits = self.block_decoder(emb)  # (B,X,Y,Z,n_blocks)
        block_ids = torch.argmax(block_logits, dim=-1)  # (B,X,Y,Z)
        
        # 预测状态标签
        state_logits = self.state_decoder(emb)  # (B,X,Y,Z,n_states)
        state_weights = torch.tanh(state_logits)
        
        # 生成状态标签 (B,X,Y,Z,N_STATE)
        adjusted_scores = torch.where(state_weights > 0, state_weights, -torch.inf)
        topk_values, topk_indices = torch.topk(adjusted_scores, k=self.N_STATE, dim=-1)
        mask_valid = topk_values != -torch.inf
        state_ids = topk_indices * mask_valid.long()
        
        # 合并结果
        output = torch.cat([
            block_ids.unsqueeze(-1), 
            state_ids
        ], dim=-1)  # (B,X,Y,Z,1+N_STATE)
        return output