import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, knn_graph

class DeepDynamicEdgeConvGnn(torch.nn.Module):
    """
    A Deep Dynamic Edge Convolution GNN with configurable depth and skip connections.

    Each layer re-computes the k-NN graph in feature space, allowing for 
    dynamic grouping of nodes as they become semantically similar.
    """
    def __init__(self, 
                 in_channels=4,      # x, y, z + frame_time
                 out_channels=1,     # valid/invalid mask
                 hidden_channels=64, # Latent dimension
                 num_layers=3,       # Number of EdgeConv layers
                 k=20,               # Number of neighbors
                 dropout=0.5,
                 **kwargs):
        super().__init__()
        self.k = k
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.input_norm = nn.BatchNorm1d(in_channels)

        # Create sequential EdgeConv layers
        self.convs = nn.ModuleList()
        
        # First layer: input -> hidden
        self.convs.append(DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(in_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ), k=k, aggr='max'))

        # Subsequent layers: hidden -> hidden
        for _ in range(num_layers - 1):
            self.convs.append(DynamicEdgeConv(
                nn=nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU()
                ), k=k, aggr='max'))

        # Classifier: Concatenate all intermediate features (Skip Connections)
        # Total dim = hidden_channels * num_layers
        total_dim = hidden_channels * num_layers
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_channels)
        )

    def forward(self, x, batch=None, return_intermediate=False, return_fused=False):
        """
        Forward pass with optional intermediate feature/graph extraction.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.input_norm(x)
        
        layer_features = []
        intermediates = {}
        
        h = x
        for i, conv in enumerate(self.convs):
            # If we need intermediates, we compute the knn_graph manually BEFORE the conv
            # because DynamicEdgeConv takes (x, batch) and does it internally.
            if return_intermediate:
                # We record the graph of the features ENTERING this layer
                edge_index = knn_graph(h, k=self.k, batch=batch, loop=False)
                intermediates[f"layer_{i}_edge_index"] = edge_index
            
            h = conv(h, batch)
            layer_features.append(h)
            
            if return_intermediate:
                intermediates[f"layer_{i}_features"] = h

        # Concatenate skip connections
        fused = torch.cat(layer_features, dim=1) # [N, hidden * num_layers]
        out = self.classifier(fused)
        
        if return_intermediate:
            intermediates["fused"] = fused
            return out, intermediates
        
        if return_fused:
            return out, fused
        
        return out
