import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.utils import to_dense_batch

class LinearAttentionLayer(nn.Module):
    """
    O(N) Linear Attention using the ELU kernel trick.
    Formula: V_out = (phi(Q) * (phi(K)^T * V)) / (phi(Q) * sum(phi(K)^T))
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def feature_map(self, x):
        return F.elu(x) + 1  # ELU kernel ensures positivity

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.feature_map(q), self.feature_map(k)
        
        # Linear Attention: Q @ (K.T @ V)
        kv = torch.matmul(k.transpose(-2, -1), v)
        z = 1 / (torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        attn_out = torch.matmul(q, kv) * z
        
        return self.out_proj(attn_out.transpose(1, 2).reshape(B, N, C))

class FourierPositionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1.0):
        super().__init__()
        self.scale = scale
        # We need to project 'in_channels' to 'out_channels'. 
        # Since we use sin & cos, we need out_channels / 2 frequencies.
        self.B = nn.Linear(in_channels, out_channels // 2, bias=False)
        # Initialize B with Gaussian to simulate random Fourier features
        nn.init.normal_(self.B.weight, std=1.0)

    def forward(self, x):
        x_proj = 2 * torch.pi * self.B(x) * self.scale
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class GlobalContextStem(nn.Module):
    """
    Runs a lightweight Linear Transformer on the full point cloud
    to generate a 'Global Context' vector for every point.
    """
    def __init__(self, in_channels, global_dim=32, layers=2, use_fourier_features=False):
        super().__init__()
        self.input_emb = nn.Linear(in_channels, global_dim)
        
        # Positional Encoding (Crucial for Transformers)
        if use_fourier_features:
             self.pos_emb = FourierPositionEncoder(3, global_dim, scale=10.0)
        else:
             self.pos_emb = nn.Linear(3, global_dim) 
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LinearAttentionLayer(global_dim),
                nn.LayerNorm(global_dim),
                nn.Sequential(
                    nn.Linear(global_dim, global_dim * 2), 
                    nn.ReLU(), 
                    nn.Linear(global_dim * 2, global_dim)
                ),
                nn.LayerNorm(global_dim)
            ]) for _ in range(layers)
        ])

    def forward(self, x, pos, batch):
        # 1. Unstack to Dense Batch [B, N, C]
        x_dense, mask = to_dense_batch(x, batch)
        pos_dense, _ = to_dense_batch(pos, batch)
        
        # 2. Embed
        h = self.input_emb(x_dense) + self.pos_emb(pos_dense)
        
        # 3. Process
        for attn, norm1, ffn, norm2 in self.layers:
            h = norm1(h + attn(h))
            h = norm2(h + ffn(h))
            
        # 4. Restack to PyG format [Total_Points, C]
        return h[mask]

class RuiyangTestModel(nn.Module):
    def __init__(self,
        hidden_channels=32,
        out_channels=1,
        k=20,
        dropout=0.5,
        use_gnn=True,
        global_dim = 32,
        use_global_context=True,
        use_fourier_features=False,
        **kwargs):
        super().__init__()
        self.k = k
        self.use_gnn = use_gnn
        self.use_global_context = use_global_context
        
        # --- 1. Norms ---
        self.spatial_norm = nn.BatchNorm1d(3)
        self.time_norm = nn.BatchNorm1d(1)

        # --- 2. Global Context Stem ---
        # Adds 'global_dim' features to every point
        if self.use_global_context:
            self.global_stem = GlobalContextStem(in_channels=4, global_dim=global_dim, layers=2, use_fourier_features=use_fourier_features)
        else:
            global_dim = 0 # No global context features

        # --- 3. Spatial Stream (Geometry) ---
        # Input: 3 (Coords)
        # Processed independently from global context
        if self.use_gnn:
            self.conv_spatial = DynamicEdgeConv(
                nn=nn.Sequential(
                    nn.Linear(3 * 2, hidden_channels), # *2 because EdgeConv concats (x_i, x_j - x_i)
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU()
                ), k=k, aggr='max')

            # --- 4. Persistence Stream (Stability) ---
            # Input: 4 (Coords+Time)
            self.conv_persistence = DynamicEdgeConv(
                nn=nn.Sequential(
                    nn.Linear(4 * 2, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU()
                ), k=k, aggr='max')

        # --- 5. Fusion & Classifier ---
        # We removed the heavy Fusion GNN as per your optimization
        # Just simple concatenation + MLP
        
        # Calculate total dimension based on active components
        total_dim = 0
        if self.use_gnn:
            total_dim += hidden_channels * 2 # Spatial + Persistence
        
        if self.use_global_context:
            total_dim += global_dim
            
        # If no features are selected (edge case), ensure at least some dimension or handle error
        # Assuming at least one is True or user knows what they are doing.
        # But if GNN is OFF, we probably want to pass the raw/global features to classifier?
        # The user said: "send the output of the attention layer directly to the final classifier".
        # So if use_gnn is False, and use_global_context is True, total_dim = global_dim.
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_channels)
        )

    def forward(self, x, batch=None):
        # x: [N, 4] -> x, y, z, normalized_frame_time
        
        # --- A. Normalization ---
        pos = x[:, :3]
        x_spatial_norm = self.spatial_norm(pos)
        
        time = x[:, 3].unsqueeze(1)
        x_time_norm = self.time_norm(time)
        
        # --- B. NEW: Global Context Extraction ---
        # Run Linear Attention on the FULL cloud
        # global_ctx: [N, 32]
        if self.use_global_context:
            global_ctx = self.global_stem(x, pos, batch)
        else:
            global_ctx = None

        # --- C. Stream Processing ---
        features_list = []
        
        if self.use_gnn:
            # Prepare Stream Inputs
            # Independent of Global Context now
            
            # Spatial Input: [Norm_XYZ]
            x_spatial_in = x_spatial_norm
            # Persistence Input: [Norm_XYZ, Norm_Time]
            x_persistence_in = torch.cat([x_spatial_norm, x_time_norm], dim=1)

            # Now the GNN knows "local neighbor distance"
            out_spatial = self.conv_spatial(x_spatial_in, batch)
            out_persistence = self.conv_persistence(x_persistence_in, batch)
            
            features_list.append(out_spatial)
            features_list.append(out_persistence)
            
        if self.use_global_context:
            features_list.append(global_ctx)

        # --- E. Final Classification ---
        # Combine: Spatial + Persistence + Global Context
        final_concat = torch.cat(features_list, dim=1)
        
        return self.classifier(final_concat)