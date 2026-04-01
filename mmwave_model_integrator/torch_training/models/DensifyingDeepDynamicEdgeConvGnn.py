import torch
import torch.nn as nn
from torch_geometric.nn import knn, fps
from torch_scatter import scatter, scatter_add
from mmwave_model_integrator.torch_training.models.DeepDynamicEdgeConvGnn import DeepDynamicEdgeConvGnn

class DensifyingDeepDynamicEdgeConvGnn(torch.nn.Module):
    """
    Densification Wrapper: Handles high-resolution inputs by interpolating 
    global context from a fixed-size sparse skeleton point cloud using IDW.
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=1,
                 hidden_channels=64, 
                 num_layers=4, 
                 k=20,
                 p=2.0, 
                 num_sparse_points=200,
                 dropout=0.5,
                 **kwargs):
        super().__init__()
        self.p = p # Power for Inverse Distance Weighting
        self.num_sparse_points = num_sparse_points
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Backbone: extracts global geometric context on the sparsified skeleton
        self.backbone = DeepDynamicEdgeConvGnn(
            in_channels=in_channels,
            out_channels=out_channels, # Unused for direct readout here, but required
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            k=k,
            dropout=dropout
        )
        
        fused_dim = hidden_channels * num_layers
        self.input_norm = nn.BatchNorm1d(in_channels)
        
        # Refinement head: Processes local point data + interpolated global context
        self.refiner = nn.Sequential(
            nn.Linear(in_channels + fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_channels)
        )

    def forward(self, x, batch=None, return_intermediate=False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1. SUBSAMPLE: Create the sparse skeleton
        # fps takes a ratio. If we want num_sparse_points total, we compute ratio.
        # This works correctly for batch_sizes > 1 as long as num_sparse_points per batch element is desired.
        num_batches = int(batch.max().item() + 1) if batch.numel() > 0 else 1
        target_sparse_points = self.num_sparse_points * num_batches
        ratio = min(1.0, float(target_sparse_points) / float(max(1, x.size(0))))

        idx_sparse = fps(x[:, :3], batch, ratio=ratio)
        x_sparse = x[idx_sparse]
        batch_sparse = batch[idx_sparse]

        intermediates = {}
        if return_intermediate:
            intermediates["idx_sparse"] = idx_sparse
            intermediates["x_sparse"] = x_sparse

        # 2. GLOBAL REASONING: Deep GNN on the sparse skeleton
        if return_intermediate:
            out_sparse, backbone_ints = self.backbone(x_sparse, batch_sparse, return_intermediate=True)
            fused_sparse = backbone_ints["fused"]
            intermediates["backbone"] = backbone_ints
            intermediates["out_sparse"] = out_sparse
        else:
            out_sparse, fused_sparse = self.backbone(x_sparse, batch_sparse, return_fused=True)

        # 3. INTERPOLATE (IDW): Densify back to all dense points
        # knn assigns each element in y (dense) to k nearest points in x (sparse)
        # Returns [2, num_dense * k] where row 0 is y (dense_idx) and row 1 is x (sparse_idx)
        assign_idx = knn(x_sparse[:, :3], x[:, :3], k=4, 
                         batch_x=batch_sparse, batch_y=batch)
        dense_idx, sparse_idx = assign_idx[0], assign_idx[1]

        # Calculate distances and weights
        dist = torch.norm(x[dense_idx, :3] - x_sparse[sparse_idx, :3], dim=-1)
        weights = 1.0 / (dist.pow(self.p) + 1e-10)
        
        # Normalize weights so they sum to 1 for each dense point
        total_weight = scatter_add(weights, dense_idx, dim=0, dim_size=x.size(0))
        normalized_weights = weights / total_weight[dense_idx]

        # Propagate features
        weighted_features = fused_sparse[sparse_idx] * normalized_weights.unsqueeze(-1)
        fused_dense = scatter_add(weighted_features, dense_idx, dim=0, dim_size=x.size(0))
        
        if return_intermediate:
            intermediates["assign_idx"] = assign_idx # Contains dense_idx (0) and sparse_idx (1)

        # 4. FINAL CLASSIFICATION: Concatenate Local + Interpolated Global
        x_norm = self.input_norm(x)
        final_input = torch.cat([x_norm, fused_dense], dim=-1)
        
        out = self.refiner(final_input)

        if return_intermediate:
            return out, intermediates
        return out
