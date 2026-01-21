import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, radius, knn_interpolate, DynamicEdgeConv
from torch_geometric.utils import scatter

class DenseInputStem(nn.Module):
    """
    The 'Compression' Layer.
    Aggregates dense local details into the selected Mid-Points before downsampling.
    """
    def __init__(self, in_channels, out_channels, k=32):
        super().__init__()
        self.k = k
        # MLP to process the dense neighbors
        # Input: [Features + Relative_Pos]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x_dense, pos_dense, pos_mid, batch_dense, batch_mid):
        # 1. For every Mid-Point, find k neighbors in the Dense cloud
        # radius returns (row=Dense_Idx, col=Mid_Idx)
        row, col = radius(pos_dense, pos_mid, r=1.0, 
                          batch_x=batch_dense, batch_y=batch_mid, 
                          max_num_neighbors=self.k)
        
        # 2. Encode Relative Geometry
        edge_pos = pos_dense[row] - pos_mid[col]
        edge_feat = torch.cat([x_dense[row], edge_pos], dim=1)
        
        # 3. Run MLP on neighbors
        encoded_feat = self.mlp(edge_feat)
        
        # 4. Aggregate (Max Pool) -> One vector per Mid-Point
        # This vector summarizes the local geometry of the dense cloud
        mid_features = scatter(encoded_feat, col, dim=0, dim_size=pos_mid.size(0), reduce='max')
        
        return mid_features


class TwoStreamDynamicBlock(nn.Module):
    """
    Runs Spatial and Persistence learning on the Mid-Level Point Cloud.
    """
    def __init__(self, in_channels, hidden_channels, k=16):
        super().__init__()
        
        # Stream 1: Spatial (x,y,z)
        self.spatial_conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(in_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ), k=k, aggr='max'
        )

        # Stream 2: Persistence (x,y,z,t)
        self.temporal_conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(in_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ), k=k, aggr='max'
        )
        
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.bn_fusion = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, batch):
        # DynamicEdgeConv uses the input 'x' to calculate distances dynamically.
        # It creates a new graph every forward pass based on feature similarity.
        x_spatial = self.spatial_conv(x, batch)
        x_temporal = self.temporal_conv(x, batch)
        
        combined = torch.cat([x_spatial, x_temporal], dim=1)
        return F.relu(self.bn_fusion(self.fusion(combined)))


class CascadeTwoStreamNet(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=1, 
                 mid_samples=512, anchor_samples=128):
        super().__init__()
        self.mid_samples = mid_samples       # e.g., 512 (Random)
        self.anchor_samples = anchor_samples # e.g., 128 (FPS from Mid)
        
        # --- Pre-processing ---
        self.input_norm = nn.BatchNorm1d(in_channels)
        self.pos_norm = nn.BatchNorm1d(3)

        # --- Stage 1: Dense -> Mid (The Stem) ---
        # "Summarize the local neighborhood before downsampling"
        self.stem = DenseInputStem(in_channels, hidden_channels, k=32)

        # --- Stage 2: Mid-Level Processing (The Brain) ---
        # "Understand geometry and persistence on the cleaner subset"
        self.mid_gnn = TwoStreamDynamicBlock(hidden_channels, hidden_channels, k=20)

        # --- Stage 3: Global Context (The Anchor) ---
        # "Look at the whole room using a Transformer"
        self.pre_transformer = nn.Linear(hidden_channels + 3, hidden_channels)
        self.global_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_channels, 
            nhead=4, 
            dim_feedforward=hidden_channels*2, 
            batch_first=True, 
            dropout=0.2
        )

        # --- Stage 4: Upsampling (The Decoder) ---
        # Global -> Mid
        self.upsample_to_mid = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels), # *2 for skip connection
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )
        
        # Mid -> Dense
        self.upsample_to_dense = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, out_channels)
        )

    def get_random_sample_idx(self, batch, num_samples):
        """Fast Random Sampling for Mid-Points."""
        batch_size = batch.max().item() + 1
        indices = []
        for i in range(batch_size):
            mask = (batch == i)
            idx = torch.nonzero(mask).squeeze()
            if idx.numel() == 0: continue
            
            if idx.numel() > num_samples:
                # Random Permutation
                perm = torch.randperm(idx.numel(), device=batch.device)[:num_samples]
                selected = idx[perm]
            else:
                # Padding with replacement
                perm = torch.randint(0, idx.numel(), (num_samples,), device=batch.device)
                selected = idx[perm]
            indices.append(selected)
        return torch.cat(indices)

    def get_fps_sample_idx(self, x_metric, batch, num_samples):
        """
        Runs FPS on an arbitrary metric tensor (Geometry, Features, or Both).
        """
        batch_size = batch.max().item() + 1
        indices = []
        for i in range(batch_size):
            mask = (batch == i)
            idx = torch.nonzero(mask).squeeze()
            
            # Extract the metric subset for this batch item
            x_subset = x_metric[idx]
            
            if idx.numel() >= num_samples:
                # FPS calculates Euclidean distance on x_subset
                # If x_subset is [Pos (3), Feat (64)], it calculates distance in 67D space.
                fps_local = fps(x_subset, ratio=1.0) 
                fps_local_topk = fps_local[:num_samples]
                selected = idx[fps_local_topk]
            else:
                perm = torch.randint(0, idx.numel(), (num_samples,), device=x_metric.device)
                selected = idx[perm]
            indices.append(selected)
        return torch.cat(indices)

    def forward(self, x, batch):
        # --- A. Setup ---
        raw_pos = x[:, :3].clone()
        norm_x = self.input_norm(x)
        
        # --- B. Stage 1: Dense -> Mid (Random Sampling) ---
        idx_mid = self.get_random_sample_idx(batch, self.mid_samples)
        pos_mid = raw_pos[idx_mid]
        batch_mid = batch[idx_mid]
        
        # Run Stem: Aggregate Dense info into Mid-Points
        # x_mid shape: [Total_Mid, Hidden]
        x_mid = self.stem(norm_x, raw_pos, pos_mid, batch, batch_mid)

        # --- C. Stage 2: Mid-Level Processing (Two-Stream) ---
        # Refine Mid-Features with Dynamic Graph
        x_mid_refined = self.mid_gnn(x_mid, batch_mid) 
        x_mid = x_mid + x_mid_refined # Residual connection

        # --- D. Stage 3: Global Processing (Feature-Aware FPS) ---
        
        # 1. Construct Sampling Metric: [Position, Features]        
        # Normalize pos locally for the sampling metric so it doesn't dominate/vanish
        # relative to the feature magnitude.
        pos_metric = pos_mid / (pos_mid.std(dim=0, keepdim=True) + 1e-6)
        feat_metric = x_mid / (x_mid.std(dim=0, keepdim=True) + 1e-6)
        
        # Concatenate: 
        # We weigh position slightly higher (1.5x) to ensure we don't collapse 
        # spatially (i.e. to prevent all anchors ending up in one distinct corner).
        sampling_metric = torch.cat([pos_metric * 1.5, feat_metric], dim=1)
        
        # 2. Run FPS on this Hybrid Metric
        # This selects indices relative to the 'pos_mid' tensor
        idx_anchor_local = self.get_fps_sample_idx(sampling_metric, batch_mid, self.anchor_samples)
        
        # 3. Gather Data
        pos_anchor = pos_mid[idx_anchor_local]
        x_anchor = x_mid[idx_anchor_local]
        
        # 4. Prepare for Transformer [B, N, C]
        B = batch.max().item() + 1
        
        # Prepare for Transformer [B, N, C]
        B = batch.max().item() + 1
        x_anchor_view = x_anchor.view(B, self.anchor_samples, -1)
        pos_anchor_view = pos_anchor.view(B, self.anchor_samples, 3)
        
        # Inject Geometry
        norm_pos_anchor = self.pos_norm(pos_anchor_view.reshape(-1, 3)).view(B, self.anchor_samples, 3)
        global_in = torch.cat([x_anchor_view, norm_pos_anchor], dim=2)
        global_in = self.pre_transformer(global_in)
        
        # Run Transformer
        x_global_view = self.global_transformer(global_in) # [B, N, C]
        
        # --- E. Stage 4: Upsampling (Decoder) ---
        
        # 1. Global -> Mid
        x_global_flat = x_global_view.reshape(-1, x_global_view.size(-1))
        
        # Create batch vector for anchors
        # Since we view()ed, we know the order is sequential blocks
        batch_anchor = torch.arange(B, device=x.device).repeat_interleave(self.anchor_samples)
        
        # Interpolate Global context back to Mid points
        ctx_mid = knn_interpolate(x_global_flat, pos_anchor, pos_mid, 
                                  batch_x=batch_anchor, batch_y=batch_mid, k=3)
        
        # Fuse Global context with Mid features
        x_mid_fused = self.upsample_to_mid(torch.cat([x_mid, ctx_mid], dim=1))
        
        # 2. Mid -> Dense
        # Interpolate Mid features back to ALL Dense points
        ctx_dense = knn_interpolate(x_mid_fused, pos_mid, raw_pos, 
                                    batch_x=batch_mid, batch_y=batch, k=3)
        
        # 3. Final Classification
        dense_input = torch.cat([norm_x, ctx_dense], dim=1)
        return self.upsample_to_dense(dense_input)