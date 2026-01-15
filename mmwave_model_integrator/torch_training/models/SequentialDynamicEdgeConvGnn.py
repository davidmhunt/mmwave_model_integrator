import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv

class SequentialDynamicEdgeConv(torch.nn.Module):
    """
    A Sequential Dynamic Edge Convolution Graph Neural Network.

    This model progressively extracts features from point cloud data using dynamic graph construction.
    It builds a k-NN graph in feature space at multiple layers to learn both local geometry and 
    higher-level semantic features.

    Architecture:
        1. Input Normalization (BatchNorm)
        2. DynamicEdgeConv Layer 1 (Geometry extraction)
        3. DynamicEdgeConv Layer 2 (Semantic feature extraction, re-computed graph)
        4. Skip Connections (Concatenation of Layer 1 & 2)
        5. MLP Classifier with Dropout
    """
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=1, k=20, dropout=0.5, **kwargs):
        """
        Initializes the SequentialDynamicEdgeConv model.

        Args:
            in_channels (int): Number of input features per point (e.g., 4 for x, y, z, time).
            hidden_channels (int): Dimension of the internal feature embeddings.
            out_channels (int): Number of output classes (e.g., 1 for probability).
            k (int): Number of nearest neighbors to consider in the dynamic graph. 
                     Standard values for point clouds are 20-30.
            dropout (float): Dropout probability used in the final classifier layer for regularization.
            **kwargs: Additional keyword arguments (unused).
        """
        super(SequentialDynamicEdgeConv, self).__init__()
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
            nn.Dropout(dropout), # Helps with your small dataset size/overfitting
            nn.Linear(128, out_channels)
        )

    def forward(self, x, batch=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input node features of shape [Num_Points, in_channels].
            batch (torch.Tensor, optional): Batch vector of shape [Num_Points] indicating 
                                            which sample each point belongs to.

        Returns:
            torch.Tensor: Output logits of shape [Num_Points, out_channels].
        """

        #1. Take the batch norm of the dataset
        x = self.input_norm(x)

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