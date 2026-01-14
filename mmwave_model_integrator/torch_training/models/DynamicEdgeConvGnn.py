import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv

class RadarDynamicClassifier(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=1, k=20):
        """
        Args:
            in_channels: 4 (x, y, z, time_decay)
            hidden_channels: Size of internal embedding
            out_channels: 1 (Probability of being valid)
            k: Number of neighbors to consider (20-30 is standard for point clouds)
        """
        super(RadarDynamicClassifier, self).__init__()
        self.k = k

        #input batch normalization (unused for now)
        self.input_norm = nn.BatchNorm1d(in_channels)

        # --- Layer 1: Input (x,y,z,t) -> 64 features ---
        # DynamicEdgeConv expects an MLP that processes: cat(x_i, x_j - x_i)
        # Input dim is therefore in_channels * 2
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(in_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ),
            k=k, aggr='max' # 'max' aggregation captures sharp features (edges/boundaries) better than 'mean'
        )

        # --- Layer 2: 64 -> 128 features ---
        # We re-compute the graph here! This allows points to group by "feature similarity"
        # rather than just physical distance.
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels * 2),
                nn.BatchNorm1d(hidden_channels * 2),
                nn.ReLU(),
                nn.Linear(hidden_channels * 2, hidden_channels * 2),
                nn.BatchNorm1d(hidden_channels * 2),
                nn.ReLU()
            ),
            k=k, aggr='max'
        )

        # --- Layer 3: Final Point-wise MLP ---
        # Deep Graph CNNs (DGCNN) typically concatenate features from all layers (skip connections)
        # combined_features = 64 (layer1) + 128 (layer2) = 192
        total_dim = hidden_channels + (hidden_channels * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5), # Helps with your small dataset size/overfitting
            nn.Linear(128, out_channels)
        )

    def forward(self, x, batch=None):
        """
        Args:
            x: [Num_Points, 4] tensor (x,y,z,time)
            batch: [Num_Points] tensor indicating which cloud the point belongs to
                   (Required for DynamicEdgeConv to not link points across different samples)
        """

        # 1. Feature Extraction (Layer 1)
        # DynamicEdgeConv computes the k-NN graph internally on the GPU
        x1 = self.conv1(x, batch)

        # 2. Feature Extraction (Layer 2)
        # The graph is dynamically re-computed based on x1 features
        x2 = self.conv2(x1, batch)

        # 3. Feature Concatenation (Skip Connections)
        # We combine low-level geometry (x1) with high-level semantics (x2)
        x_cat = torch.cat([x1, x2], dim=1)

        # 4. Final Classification
        out = self.classifier(x_cat)
        
        # Note: We return logits (raw scores). 
        # Use BCEWithLogitsLoss in your training loop for numerical stability.
        return out