import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockEncoder(nn.Module):
    def __init__(self, n_blocks, n_states, n_state_channels, embed_dim):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_states = n_states
        self.n_state_channels = n_state_channels
        self.embed_dim = embed_dim
        
        # Block type embedding
        self.block_embed = nn.Embedding(n_blocks, embed_dim)
        
        # State embeddings (one per state channel)
        self.state_embeds = nn.ModuleList([
            nn.Embedding(n_states, embed_dim) 
            for _ in range(n_state_channels)
        ])
        
        # Feature compression
        self.compressor = nn.Conv3d(
            embed_dim * (1 + n_state_channels), 
            embed_dim, 
            kernel_size=1
        )

    def forward(self, x):
        # x: (B,X,Y,Z,1+N_STATE)
        block_ids = x[..., 0].long()  # (B,X,Y,Z)
        state_ids = x[..., 1:].long()  # (B,X,Y,Z,N_STATE)
        
        # Block embedding
        block_emb = self.block_embed(block_ids)  # (B,X,Y,Z,E)
        
        # State embeddings
        state_embs = []
        for i in range(self.n_state_channels):
            emb = self.state_embeds[i](state_ids[..., i])  # (B,X,Y,Z,E)
            state_embs.append(emb)
        
        # Concatenate all features
        combined = torch.cat([block_emb] + state_embs, dim=-1)  # (B,X,Y,Z,E*(1+N))
        combined = combined.permute(0,4,1,2,3)  # (B,C,X,Y,Z)
        
        # Compress to embed_dim
        out = self.compressor(combined)  # (B,E,X,Y,Z)
        return out.permute(0,2,3,4,1)  # (B,X,Y,Z,E)

class BlockDecoder(nn.Module):
    def __init__(self, n_blocks, n_states, n_state_channels, embed_dim):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_states = n_states
        self.n_state_channels = n_state_channels
        self.embed_dim = embed_dim
        
        # Feature decompression
        self.decompressor = nn.Conv3d(
            embed_dim,
            embed_dim * (1 + n_state_channels),
            kernel_size=1
        )
        
        # Prediction heads
        self.block_head = nn.Linear(embed_dim, n_blocks)
        self.state_heads = nn.ModuleList([
            nn.Linear(embed_dim, n_states)
            for _ in range(n_state_channels)
        ])

    def forward(self, x):
        # x: (B,X,Y,Z,E)
        B, X, Y, Z, E = x.shape
        
        # Decompress features
        x = x.permute(0,4,1,2,3)  # (B,E,X,Y,Z)
        decompressed = self.decompressor(x)  # (B,E*(1+N),X,Y,Z)
        decompressed = decompressed.permute(0,2,3,4,1)  # (B,X,Y,Z,E*(1+N))
        decompressed = decompressed.view(B, X, Y, Z, 1+self.n_state_channels, E)
        
        # Split features
        block_feats = decompressed[..., 0, :]  # (B,X,Y,Z,E)
        state_feats = decompressed[..., 1:, :]  # (B,X,Y,Z,N,E)
        
        # Predict block types
        block_logits = self.block_head(block_feats)  # (B,X,Y,Z,n_blocks)
        
        # Predict state values
        state_logits = []
        for i in range(self.n_state_channels):
            logits = self.state_heads[i](state_feats[..., i, :])  # (B,X,Y,Z,n_states)
            state_logits.append(logits)
        
        # Combine outputs
        state_logits = torch.stack(state_logits, dim=-1)  # (B,X,Y,Z,n_states,N)
        return torch.cat([
            block_logits.unsqueeze(-1),
            state_logits
        ], dim=-1)  # (B,X,Y,Z,1+N,n_states)