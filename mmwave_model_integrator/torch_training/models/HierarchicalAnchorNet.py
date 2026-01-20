import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, radius, knn_interpolate
from torch_geometric.utils import scatter

class HybridLocalAggregator(nn.Module):
    """Aggregates local graph features using a dual-path strategy.
    
    This module combines the geometric sharpness of Max Pooling with the 
    optional selective filtering of Additive Attention (GAT-style). It assumes 
    the input features `x` serve as both the signal and the coordinate space 
    for calculating relative differences.

    Attributes:
        use_attention (bool): Flag to enable/disable the attention branch.
        input_norm (nn.BatchNorm1d): Batch normalization for input features.
        mlp (nn.Sequential): Shared Multi-Layer Perceptron for edge encoding.
        attention_predictor (nn.Linear, optional): Layer to compute attention scores.
        fusion_layer (nn.Linear): Linear layer to fuse the aggregated pathways.
        output_norm (nn.BatchNorm1d): Batch normalization for the output.
    """

    def __init__(self, in_channels: int, hidden_channels: int, use_attention: bool = True):
        """Initializes the HybridLocalAggregator.

        Args:
            in_channels (int): The number of input features per node.
            hidden_channels (int): The dimension of the internal feature embeddings.
            use_attention (bool, optional): If True, enables the secondary Attention 
                pathway for noise filtering. Defaults to True.
        """
        super().__init__()
        self.use_attention = use_attention
        
        # 1. Input Normalization
        self.input_norm = nn.BatchNorm1d(in_channels)
        
        # 2. Shared Feature Encoder
        # Input dimension is in_channels * 2 because we concatenate:
        # [Source_Features (N) || (Source - Target) Difference (N)]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )
        
        # 3. Dual-Path Components
        if self.use_attention:
            # Path B: Attention Weight Predictor
            # Learns a scalar score (0-1) for each edge based on feature importance
            self.attention_predictor = nn.Linear(hidden_channels, 1)
            
            # Fusion: Compresses [Max_Path (H) + Attn_Path (H)] -> H
            self.fusion_layer = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            # Single Path Fusion: Just processes [Max_Path (H)] -> H
            self.fusion_layer = nn.Linear(hidden_channels, hidden_channels)

        self.output_norm = nn.BatchNorm1d(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, dim_size: int = None) -> torch.Tensor:
        """Performs the aggregation pass.

        Args:
            x (torch.Tensor): Dense node features of shape [N, in_channels].
            edge_index (torch.Tensor): Graph connectivity of shape [2, E], 
                where row=Source indices and col=Target indices.
            dim_size (int, optional): The number of target nodes (anchors). 
                If None, inferred from `x` or `edge_index`.

        Returns:
            torch.Tensor: Aggregated node features of shape [dim_size, hidden_channels].
        """
        # Normalize inputs immediately
        x = self.input_norm(x)

        # Unpack graph connectivity
        row, col = edge_index
        
        # Determine output dimension size (number of anchors)
        if dim_size is None:
            dim_size = x.size(0) if x is not None else col.max().item() + 1

        # --- A. Shared Edge Feature Computation ---
        # Calculate relative features: (Source - Target)
        # This genericizes "Relative Position" to any feature space provided in x
        edge_relative_features = x[row] - x[col]
        
        # Concatenate: [Neighbor Features] + [Relative Difference]
        edge_input = torch.cat([x[row], edge_relative_features], dim=1)
        
        # Compute Edge Embeddings (Messages) -> [Num_Edges, Hidden]
        edge_embeddings = self.mlp(edge_input)
        
        # --- B. Path 1: Max Pooling (Geometry Preservation) ---
        # Captures extreme values (corners, sharp edges)
        # Always active as it provides the robust structural signal
        out_max = scatter(edge_embeddings, col, dim=0, dim_size=dim_size, reduce='max')
        
        if self.use_attention:
            # --- C. Path 2: Attention (Noise Filtering) ---
            # Learns to down-weight noise points and up-weight consistent surfaces
            alpha_logits = self.attention_predictor(edge_embeddings)
            
            # Normalize weights per-anchor (Softmax over neighbors)
            alpha = scatter(alpha_logits, col, dim=0, dim_size=dim_size, reduce='softmax')
            
            # Weighted Sum aggregation
            out_att = scatter(edge_embeddings * alpha, col, dim=0, dim_size=dim_size, reduce='sum')
            
            # --- D. Dual-Path Fusion ---
            combined = torch.cat([out_max, out_att], dim=1)
            out = self.fusion_layer(combined)
        else:
            # Single Path Pass-through
            out = self.fusion_layer(out_max)

        return F.relu(self.output_norm(out))


class HierarchicalAnchorNet(torch.nn.Module):
    """Hierarchical Spatio-Temporal GNN for Radar Ghost Filtering.
    
    This architecture implements a Factorized "Anchor-Probe" strategy:
    1. Selects a sparse set of Anchors using FPS.
    2. Aggregates context using two parallel streams (Spatial vs Temporal).
    3. Fuses streams and performs global reasoning via Transformer.
    4. Broadcasts Anchor validity scores back to dense points.

    Attributes:
        input_norm (nn.BatchNorm1d): Initial normalization for raw input features.
        pos_norm (nn.BatchNorm1d): Explicit normalization for coordinates.
        spatial_aggregator (HybridLocalAggregator): Encodes geometric shape.
        temporal_aggregator (HybridLocalAggregator): Encodes persistence.
        stream_fusion (nn.Linear): Fuses the two streams.
        global_transformer (nn.TransformerEncoderLayer): Models room-scale context.
        final_classifier (nn.Sequential): Classifies point validity.
    """

    def __init__(self, in_channels: int = 4, hidden_channels: int = 64, 
                 out_channels: int = 1, num_anchors: int = 128, 
                 use_attention_aggregator: bool = True, **kwargs):
        """Initializes the HierarchicalAnchorNet.

        Args:
            in_channels (int): Input feature dimension. Assumes first 3 are XYZ.
                               Remaining are Time/Doppler/etc.
            hidden_channels (int): Internal embedding dimension.
            out_channels (int): Output classes (1 for binary classification).
            num_anchors (int): Number of sparse anchors to sample.
            use_attention_aggregator (bool): Whether to use Attention in aggregators.
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.k_local = 32  # Neighbor count for local aggregation
        
        # --- Stage 0: Pre-processing ---
        self.input_norm = nn.BatchNorm1d(in_channels)
        self.pos_norm = nn.BatchNorm1d(3) # For Transformer injection
        
        # Calculate split dimensions
        # Assuming X,Y,Z are first 3 channels.
        spatial_dim = 3 
        temporal_dim = in_channels - 3 # e.g. 1 if input is just X,Y,Z,T
        
        # We split the hidden capacity between the two streams
        stream_hidden = hidden_channels // 2
        
        # --- Stage 1: Parallel Local Aggregation ---
        
        # Stream A: Spatial (Geometry)
        # Inputs: X, Y, Z
        self.spatial_aggregator = HybridLocalAggregator(
            in_channels=spatial_dim, 
            hidden_channels=stream_hidden,
            use_attention=use_attention_aggregator
        )
        
        # Stream B: Temporal (Persistence)
        # Inputs: T (and optionally Doppler/Intensity)
        self.temporal_aggregator = HybridLocalAggregator(
            in_channels=temporal_dim, 
            hidden_channels=stream_hidden,
            use_attention=use_attention_aggregator
        )
        
        # Fusion Layer (Concatenates Stream A + Stream B back to hidden_channels)
        self.stream_fusion = nn.Linear(stream_hidden * 2, hidden_channels)
        self.bn_fusion = nn.BatchNorm1d(hidden_channels)

        # --- Stage 2: Global Context (Sparse -> Sparse) ---
        # Transformer input = Fused Features + Normalized XYZ (3)
        self.global_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_channels + 3, 
            nhead=4, 
            dim_feedforward=hidden_channels * 2,
            batch_first=True,
            dropout=0.2
        )

        # --- Stage 3: Final Point Classification (Dense) ---
        # Combines Raw Features + Interpolated Anchor Context
        self.final_classifier = nn.Sequential(
            nn.Linear((hidden_channels + 3) + in_channels, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, out_channels)
        )

    def get_fixed_random_anchors(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Samples exactly self.num_anchors random points for each batch item.
        Fast, stochastic, good for dense clouds.
        """
        batch_size = batch.max().item() + 1
        selected_indices = []
        
        for i in range(batch_size):
            # 1. Get indices for current frame
            mask = (batch == i)
            indices = torch.nonzero(mask).squeeze()
            num_points = indices.numel()
            
            # 2. Sample
            if num_points >= self.num_anchors:
                # No replacement (Standard random subset)
                perm = torch.randperm(num_points, device=pos.device)[:self.num_anchors]
                selected = indices[perm]
            else:
                # Replacement (Duplicate points to fill size)
                perm = torch.randint(0, num_points, (self.num_anchors,), device=pos.device)
                selected = indices[perm]
                
            selected_indices.append(selected)
            
        return torch.cat(selected_indices)

    def get_fixed_fps_anchors(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Samples exactly self.num_anchors points using Farthest Point Sampling.
        Slower, deterministic, guarantees maximum spatial coverage.
        """
        batch_size = batch.max().item() + 1
        selected_indices = []
        
        for i in range(batch_size):
            # 1. Get data for current frame
            mask = (batch == i)
            indices = torch.nonzero(mask).squeeze()
            pos_i = pos[indices]
            num_points = indices.numel()
            
            # 2. Run FPS
            if num_points >= self.num_anchors:
                # We ask for ratio=1.0 to get the full permutation sorted by distance
                # Then we slice the top K points. This effectively gives us the K farthest points.
                local_fps_idx = fps(pos_i, ratio=1.0)
                
                # Take top K
                keep_idx = local_fps_idx[:self.num_anchors]
                selected = indices[keep_idx]
            else:
                # Corner case: Not enough points for FPS. 
                # Take all available (via FPS sort) + fill rest with random duplicates
                local_fps_idx = fps(pos_i, ratio=1.0)
                
                # Part 1: All real points
                real_selected = indices[local_fps_idx]
                
                # Part 2: Duplicates to fill the gap
                needed = self.num_anchors - num_points
                rand_fill = torch.randint(0, num_points, (needed,), device=pos.device)
                fill_selected = indices[rand_fill]
                
                selected = torch.cat([real_selected, fill_selected])

            selected_indices.append(selected)
            
        return torch.cat(selected_indices)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        
        # --- Step 0: Data Prep ---
        raw_pos = x[:, :3].clone() 
        norm_x = self.input_norm(x)
        norm_x_spatial = norm_x[:, :3] 
        norm_x_temporal = norm_x[:, 3:] 
        
        # --- Step 1: Anchor Selection ---
        # TOGGLE HERE: Switch between random and fps
        anchor_idx = self.get_fixed_random_anchors(raw_pos, batch) 
        # anchor_idx = self.get_fixed_fps_anchors(raw_pos, batch)
        
        anchor_pos = raw_pos[anchor_idx]
        anchor_batch = batch[anchor_idx]
        
        # --- Step 2: Parallel Local Aggregation ---
        row, col = radius(raw_pos, anchor_pos, r=1.0, batch_x=batch, batch_y=anchor_batch, max_num_neighbors=self.k_local)
        
        spatial_features = self.spatial_aggregator(
            x=norm_x_spatial, edge_index=(row, col), dim_size=anchor_pos.size(0)
        )
        temporal_features = self.temporal_aggregator(
            x=norm_x_temporal, edge_index=(row, col), dim_size=anchor_pos.size(0)
        )
        
        combined_streams = torch.cat([spatial_features, temporal_features], dim=1)
        anchor_features = F.relu(self.bn_fusion(self.stream_fusion(combined_streams)))

        # --- Step 3: Global Reasoning (Simplified View) ---
        norm_anchor_pos = self.pos_norm(anchor_pos)
        global_input = torch.cat([anchor_features, norm_anchor_pos], dim=1)
        
        # Safe reshape because we guaranteed exactly num_anchors per batch item
        batch_size = batch.max().item() + 1
        dense_input = global_input.view(batch_size, self.num_anchors, -1)
        
        # Transformer
        anchor_global_dense = self.global_transformer(dense_input)
        
        # Flatten back
        anchor_global = anchor_global_dense.view(-1, anchor_global_dense.size(-1))
        
        # --- Step 4: Broadcasting ---
        interpolated_context = knn_interpolate(
            anchor_global, anchor_pos, raw_pos, 
            batch_x=anchor_batch, batch_y=batch, k=3
        )
        
        dense_input = torch.cat([norm_x, interpolated_context], dim=1)
        
        return self.final_classifier(dense_input)