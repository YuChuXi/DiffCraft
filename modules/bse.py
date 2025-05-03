import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockEncoder(nn.Module):
    """
    输入形状: (B, X, Y, Z), (B, X, Y, Z, max_n_state), (B, X, Y, Z)
    输出形状: (B, X, Y, Z, E)
    """

    def __init__(self, n_blocks, n_states, embed_dim):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_states = n_states
        self.embed_dim = embed_dim

        self.block_embed = nn.Embedding(n_blocks, embed_dim)
        self.state_embed = nn.Embedding(n_states, embed_dim)

    def forward(self, block_ids, state_ids, mask=None):
        B, X, Y, Z = block_ids.shape

        # 处理Block嵌入
        if mask is not None:
            valid_block_mask = mask
            valid_block_indices = torch.nonzero(valid_block_mask, as_tuple=True)
            valid_block_ids = block_ids[valid_block_indices]
            block_emb = torch.zeros(B, X, Y, Z, self.embed_dim, device=block_ids.device)
            if len(valid_block_ids) > 0:
                valid_block_emb = self.block_embed(valid_block_ids)
                block_emb[valid_block_indices] = valid_block_emb
        else:
            block_emb = self.block_embed(block_ids)

        # 处理State嵌入
        state_emb = torch.zeros_like(block_emb)
        if mask is not None:
            valid_state_mask = mask.unsqueeze(-1) & (state_ids != 0)
        else:
            valid_state_mask = state_ids != 0

        valid_indices = torch.nonzero(valid_state_mask, as_tuple=True)
        if len(valid_indices[0]) > 0:
            valid_state_ids = state_ids[valid_indices]
            valid_state_emb = self.state_embed(valid_state_ids)

            # 计算线性索引进行累加
            b, x, y, z = valid_indices[:4]
            linear_idx = b * X * Y * Z + x * Y * Z + y * Z + z
            state_emb_flat = state_emb.view(-1, self.embed_dim)
            state_emb_flat.index_add_(0, linear_idx, valid_state_emb)
            state_emb = state_emb_flat.view_as(state_emb)

        # 处理全零状态的位置
        if mask is not None:
            valid_positions = mask
        else:
            valid_positions = torch.ones_like(block_ids, dtype=torch.bool)

        state_zero_mask = valid_positions & (state_ids.amax(dim=-1) == 0)
        if state_zero_mask.any():
            zero_emb = self.state_embed(torch.tensor(0, device=block_ids.device))
            state_emb[state_zero_mask] += zero_emb

        # 合并嵌入并应用mask
        total_emb = block_emb + state_emb
        if mask is not None:
            total_emb = total_emb * mask.unsqueeze(-1)

        return total_emb


class BlockDecoder(nn.Module):
    """
    输入形状: (B, X, Y, Z, E), (B, X, Y, Z)
    输出形状: (B, X, Y, Z, n_blocks), (B, X, Y, Z, n_states)
    """

    def __init__(self, n_blocks, n_states, embed_dim, big=False):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_states = n_states

        if big:
            self.block_decoder = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.ReLU(), nn.Linear(512, n_blocks)
            )
            self.state_decoder = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.ReLU(), nn.Linear(512, n_states)
            )
        else:
            self.block_decoder = nn.Linear(embed_dim, n_blocks)
            self.state_decoder = nn.Linear(embed_dim, n_states)

    def forward(self, emb, mask=None):
        B, X, Y, Z, E = emb.shape

        if mask is not None:
            valid_indices = torch.nonzero(mask, as_tuple=True)
            if len(valid_indices[0]) == 0:
                return (
                    torch.zeros(B, X, Y, Z, self.n_blocks, device=emb.device),
                    torch.zeros(B, X, Y, Z, self.n_states, device=emb.device),
                )
            valid_emb = emb[valid_indices]

            block_logits = torch.zeros(B, X, Y, Z, self.n_blocks, device=emb.device)
            state_logits = torch.zeros(B, X, Y, Z, self.n_states, device=emb.device)

            # 处理方块预测
            valid_block = self.block_decoder(valid_emb)
            block_logits[valid_indices] = valid_block

            # 处理状态预测（带sigmoid）
            valid_state = torch.sigmoid(self.state_decoder(valid_emb))
            state_logits[valid_indices] = valid_state
        else:
            block_logits = self.block_decoder(emb)
            state_logits = torch.sigmoid(self.state_decoder(emb))

        return block_logits, state_logits
