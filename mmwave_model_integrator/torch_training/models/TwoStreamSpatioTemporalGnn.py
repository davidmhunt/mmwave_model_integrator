import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv

class TwoStreamSpatioTemporalGnn(torch.nn.Module):
    """
    A Two-Stream Spatio-Temporal Graph Neural Network for radar point cloud processing.

    This model explicitly separates spatial geometry from temporal persistence using two parallel streams:
    1. Spatial Stream: Processes static 3D geometry (x, y, z) to detect shapes.
    2. Persistence Stream: Processes 4D data (x, y, z, t) to detect stable features over time.

    The streams are fused later to make a final classification decision.
    """
    def __init__(self,
        hidden_channels=32,
        out_channels=1,
        k=20,
        dropout=0.5,
        **kwargs):
        """
        Initializes the TwoStreamSpatioTemporalGnn model.

        Args:
            hidden_channels (int): Dimension of internal feature embeddings.
            out_channels (int): Number of output classes (e.g., 1 for binary classification).
            k (int): Number of neirest neighbors. 20 is typically optimal for capturing local planar structures.
            dropout (float): Dropout probability for regularization.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        self.k = k

        # --- 1. Robust Input Normalization ---
        # We split normalization because Space and Time have different physics.
        # BatchNorm allows the model to learn the optimal scaling ratio between 
        # "1 meter" and "1 second" automatically.
        self.spatial_norm = nn.BatchNorm1d(3) # Normalizes x,y,z
        self.time_norm = nn.BatchNorm1d(1)    # Normalizes time

        # --- 2. Spatial Stream (Geometry) ---
        # Input: 3 spatial coords * 2 (EdgeConv) = 6
        # Task: Detect "Wall-like" or "Corner-like" shapes
        self.conv_spatial = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(6, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ), k=k, aggr='max')

        # --- 3. Persistence Stream (Stability) ---
        # Input: 4 coords (x,y,z,t) * 2 = 8
        # Task: Detect points that stay in the same place over time.
        self.conv_persistence = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(8, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ), k=k, aggr='max')

        # --- 4. Fusion Layer ---
        # Combines Geometry (Is it a wall?) + Persistence (Has it been here a while?)
        fusion_dim = hidden_channels * 2
        self.conv_fusion = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU()
            ), k=k, aggr='max')

        # --- 5. Classifier ---
        # Concatenate all features: Spatial + Persistence + Fused
        total_dim = hidden_channels + hidden_channels + fusion_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_channels)
        )

    def forward(self, x, batch=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [Num_Points, 4]. Columns: x, y, z, normalized_frame_time.
            batch (torch.Tensor, optional): Batch vector indicating sample membership.

        Returns:
            torch.Tensor: Output logits.
        """
        # x: [N, 4] -> x, y, z, normalized_frame_time
        
        # --- A. Split & Normalize ---
        # Spatial: Normalize x,y,z to be roughly unit variance
        pos = x[:, :3]
        x_spatial_norm = self.spatial_norm(pos)
        
        # Time: Normalize time (critical for the persistence stream)
        time = x[:, 3].unsqueeze(1)
        x_time_norm = self.time_norm(time)
        
        # Create the 4D input for the persistence stream
        x_persistence_input = torch.cat([x_spatial_norm, x_time_norm], dim=1)

        # --- B. Stream Processing ---
        # Stream 1: Finds geometric shapes (walls, desks)
        # It ignores time, so it effectively "collapses" the video into a single 3D scan.
        out_spatial = self.conv_spatial(x_spatial_norm, batch)

        # Stream 2: Finds stable clusters
        # By using x,y,z,t, it will only connect points that are close in space AND time.
        # Transient noise (random blips) will likely lack neighbors in this 4D space.
        out_persistence = self.conv_persistence(x_persistence_input, batch)

        # --- C. Fusion & Output ---
        combined = torch.cat([out_spatial, out_persistence], dim=1)
        out_fused = self.conv_fusion(combined, batch)

        final_concat = torch.cat([out_spatial, out_persistence, out_fused], dim=1)
        
        return self.classifier(final_concat)