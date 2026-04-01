import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, fps, knn_graph
from mmwave_model_integrator.torch_training.models.DeepDynamicEdgeConvGnn import DeepDynamicEdgeConvGnn

class MacroReasoningGnn(torch.nn.Module):
    def __init__(self, 
                 in_channels=4,      # x, y, z + frame_time
                 out_channels=1,     # valid/invalid mask
                 hidden_channels=64, # Latent dimension
                 num_super_nodes=128, 
                 k=20,
                  num_heads=4,
                  use_cylindrical_encoding=False,
                  super_node_layers=1,
                  super_node_k=16,
                  **kwargs):
        super().__init__()
        self.super_k = 16
        self.k = k
        self.m = num_super_nodes 
        self.use_cylindrical_encoding = use_cylindrical_encoding
        self.hidden_channels = hidden_channels

        # --- 0. POSITIONAL ENCODER ---
        if self.use_cylindrical_encoding:
            # Lift cylindrical coords (r, sin_theta, cos_theta, z) into high-dim space
            self.pos_encoder = nn.Sequential(
                nn.Linear(4, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            local_in = (in_channels + hidden_channels) * 2
            # REFINED: Super-node GNN now uses ONLY positional embeddings to build a pure geometric map
            super_in = hidden_channels
        else:
            local_in = in_channels * 2
            super_in = hidden_channels

        # --- 1. LOCAL GEOMETRY STREAM ---
        self.local_conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(local_in, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ), k=k, aggr='max')

        # --- 2. MACRO-REASONING STREAM (The Room's Skeleton) ---
        self.super_node_layers = super_node_layers
        self.super_node_gnn = DeepDynamicEdgeConvGnn(
            in_channels=super_in, 
            out_channels=1, # This classifier is unused here but required by constructor
            hidden_channels=hidden_channels,
            num_layers=super_node_layers,
            k=super_node_k
        )
        super_context_dim = hidden_channels * super_node_layers

        # --- 3. INDIVIDUALIZED CONTEXT (Cross-Attention) ---
        # Query projection ensures h_local matches the multi-layer super-node context dimension
        self.query_proj = nn.Linear(hidden_channels, super_context_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=super_context_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

        # --- 4. CLASSIFICATION HEAD ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + super_context_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, batch=None, return_intermediate=False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Phase 0: Positional Embedding
        pos_cart = x[:, :3]
        if self.use_cylindrical_encoding:
            r = torch.norm(pos_cart[:, :2], dim=-1, keepdim=True)
            theta = torch.atan2(pos_cart[:, 1], pos_cart[:, 0]).unsqueeze(-1)

            #re-eight the angle to be more costly than ranges
            r_scale_factor = 1.0 #0.05
            theta_scale_factor = 1.0 #2.0
            
            pos_cyl = torch.cat([r * r_scale_factor,
                                torch.sin(theta) * theta_scale_factor, 
                                torch.cos(theta) * theta_scale_factor, 
                                pos_cart[:, 2:3]], dim=-1)
            pos_emb = self.pos_encoder(pos_cyl) # [N, hidden]
        
        # Phase 1: Local Representation
        if self.use_cylindrical_encoding:
            h_local_input = torch.cat([x, pos_emb], dim=-1)
            h_local = self.local_conv(h_local_input, batch)
        else:
            h_local = self.local_conv(x, batch) 

        # Phase 2: Super-Node Extraction & Skeleton Building
        indices = fps(pos_cart, batch, ratio=self.m / x.size(0))
        batch_super = batch[indices]
        
        # Phase 3: Macro-Reasoning on Pure Geometric Super-Nodes
        if self.use_cylindrical_encoding:
            h_super_raw = pos_emb[indices]
        else:
            h_super_raw = h_local[indices]
            
        # Get fused features from the DeepDynamicEdgeConvGnn
        # The internal classifier output is ignored here.
        h_super_logits, h_super_out = self.super_node_gnn(
            h_super_raw, batch_super, 
            return_intermediate=return_intermediate,
            return_fused=True
        )
        
        if return_intermediate:
            h_super_context = h_super_out["fused"]
        else:
            h_super_context = h_super_out # [M, hidden * layers]

        # Phase 4: Individualized Cross-Attention
        if self.use_cylindrical_encoding:
            # Query needs projection to match the context dimension
            q = self.query_proj(h_local + pos_emb).unsqueeze(0)
        else:
            q = self.query_proj(h_local).unsqueeze(0) 
            
        k_v = h_super_context.unsqueeze(0) # [1, M, hidden * layers]
        
        attn_output, attn_weights = self.cross_attn(
            q, k_v, k_v, 
            need_weights=return_intermediate
        )
        h_context = attn_output.squeeze(0) # [N, hidden * layers]

        # Phase 5: Final Decision
        # h_local: [N, hidden], h_context: [N, hidden * layers]
        fused = torch.cat([h_local, h_context], dim=1) 
        out = self.classifier(fused)
        
        if return_intermediate:
            return out, {
                "h_local": h_local,
                "indices": indices,
                "h_super_context": h_super_context,
                "h_context": h_context,
                "attn_weights": attn_weights.squeeze(0) if attn_weights is not None else None,
                "super_edge_index": knn_graph(h_super_raw, k=self.super_node_gnn.k, batch=batch_super),
                "super_node_intermediates": h_super_out # Contains "layer_X_features" and "layer_X_edge_index"
            }
        return out
