import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, fps

class CrossAttentionGnn(torch.nn.Module):
    def __init__(self, 
                 in_channels=4,      # x, y, z + frame_time
                 out_channels=1,     # valid/invalid mask
                 hidden_channels=64, # Latent dimension
                 num_super_nodes=128, 
                 k=20,
                 num_heads=4,
                 **kwargs):
        super().__init__()
        self.k = k
        self.m = num_super_nodes 

        # --- 1. LOCAL GEOMETRY STREAM ---
        # DynamicEdgeConv processes pairs (x_i, x_j - x_i), 
        # so the MLP input is always in_channels * 2
        self.local_conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(in_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ), k=k, aggr='max')

        # --- 2. INDIVIDUALIZED CONTEXT (Cross-Attention) ---
        # Query/Key/Value all exist in the hidden_channels space
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels, 
            num_heads=num_heads, 
            batch_first=True
        )

        # --- 3. CLASSIFICATION HEAD ---
        # We concatenate local (hidden) + context (hidden) = hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, batch=None, return_intermediate=False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Phase 1: Local Representation [N, in_channels] -> [N, hidden_channels]
        h_local = self.local_conv(x, batch) 

        # Phase 2: Super-Node Extraction
        # We sample based on spatial coordinates (first 3 columns of x)
        indices = fps(x[:, :3], batch, ratio=self.m / x.size(0))
        h_super = h_local[indices].unsqueeze(0) # [1, M, hidden]
        
        # Phase 3: Cross-Attention Context
        q = h_local.unsqueeze(0) # [1, N, hidden]
        # attn_output: [1, N, hidden]
        attn_output, attn_weights = self.cross_attn(
            q, h_super, h_super, 
            need_weights=return_intermediate
        )
        h_context = attn_output.squeeze(0) 

        # Phase 4: Final Decision
        # Concatenate Local + Global Context
        fused = torch.cat([h_local, h_context], dim=1) # [N, hidden * 2]
        out = self.classifier(fused)
        
        if return_intermediate:
            return out, {
                "h_local": h_local,
                "indices": indices,
                "h_context": h_context,
                "attn_weights": attn_weights.squeeze(0) if attn_weights is not None else None
            }
        return out