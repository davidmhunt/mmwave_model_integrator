import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, fps

def cylindrical_serialize(pos_cyl):
    """
    pos_cyl: [N, 4] -> (r, sin_theta, cos_theta, z)
    Returns: sort_indices
    """
    # 1. Calculate raw theta for sorting
    theta = torch.atan2(pos_cyl[:, 1], pos_cyl[:, 2])
    r = pos_cyl[:, 0]
    
    # 2. Lexicographical sort: Primary = theta, Secondary = r
    # This groups 'Rays' together in the sequence
    sort_keys = theta * 1000 + r # Simple weighted sum for sorting
    sort_indices = torch.argsort(sort_keys)
    
    return sort_indices

class AsymmetricShadowAttention(nn.Module):
    def __init__(self, channels, num_heads=4, temperature=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.d_k = channels // num_heads
        
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        # Bias MLP: Input (r_j - r_i, r_i)
        self.bias_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads) # Bias per head
        )
        
        # Normalize scores before softmax
        self.score_norm = nn.LayerNorm(num_heads)

    def forward(self, x, r_coords):
        """
        x: [B, P, C] (Batched Patches)
        r_coords: [B, P, 1] (Radial Coordinates)
        """
        B, P, C = x.shape
        H = self.num_heads
        D = self.d_k
        
        # Q, K, V projections
        q = self.q_proj(x).view(B, P, H, D).transpose(1, 2) # [B, H, P, D]
        k = self.k_proj(x).view(B, P, H, D).transpose(1, 2) # [B, H, P, D]
        v = self.v_proj(x).view(B, P, H, D).transpose(1, 2) # [B, H, P, D]
        
        # Scaled Dot-Product
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5) # [B, H, P, P]
        
        # --- Asymmetric Shadow Bias ---
        # Compute relative radial distances: r_j - r_i
        # r_coords is [B, P, 1]
        r_i = r_coords.unsqueeze(2) # [B, P, 1, 1]
        r_j = r_coords.unsqueeze(1) # [B, 1, P, 1]
        dist_r = r_j - r_i # [B, P, P, 1]
        
        # Input to MLP: (dist_r, r_i)
        # We need to broadcast r_i to every j
        r_i_expanded = r_i.expand(-1, -1, P, -1) # [B, P, P, 1]
        bias_input = torch.cat([dist_r, r_i_expanded], dim=-1) # [B, P, P, 2]
        
        # B_ij: [B, P, P, H]
        bias = self.bias_mlp(bias_input)
        
        # Add bias to scores and normalize
        # Transpose bias from [B, P, P, H] to [B, H, P, P]
        bias = bias.permute(0, 3, 1, 2)
        
        scores = scores + bias
        
        # Normalize scores to keep them stable
        # We apply LayerNorm over the head dimension for each pair (i, j)
        # But LayerNorm expects [..., head_dim]
        scores = scores.permute(0, 2, 3, 1) # [B, P, P, H]
        scores = self.score_norm(scores)
        scores = scores.permute(0, 3, 1, 2) # [B, H, P, P]
        
        # Sharpen with Temperature
        attn_weights = torch.softmax(scores / self.temperature, dim=-1)
        
        # Output calculation
        out = torch.matmul(attn_weights, v) # [B, H, P, D]
        out = out.transpose(1, 2).reshape(B, P, C)
        out = self.out_proj(out)
        
        return out, attn_weights

class PTv3Block(nn.Module):
    def __init__(self, channels, patch_size=16, num_heads=4, temperature=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.attn = AsymmetricShadowAttention(channels, num_heads, temperature)
        self.ln = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )

    def forward(self, x, r_coords):
        """
        x: [N, channels] (already sorted)
        r_coords: [N, 1] (radial coords, already sorted)
        """
        num_points = x.shape[0]
        num_patches = (num_points + self.patch_size - 1) // self.patch_size
        
        # 1. Padding to fit patch size
        pad_len = num_patches * self.patch_size - num_points
        if pad_len > 0:
            x_padded = torch.cat([x, torch.zeros(pad_len, x.shape[1], device=x.device)], dim=0)
            r_padded = torch.cat([r_coords, torch.zeros(pad_len, 1, device=x.device)], dim=0)
        else:
            x_padded = x
            r_padded = r_coords
            
        # 2. Reshape to [Num_Patches, Patch_Size, Channels]
        x_patches = x_padded.view(num_patches, self.patch_size, -1)
        r_patches = r_padded.view(num_patches, self.patch_size, 1)
        
        # 3. Intra-Patch Attention (Occlusion Reasoning happens here!)
        attn_out, attn_weights_patch = self.attn(x_patches, r_patches)
        x_patches = self.ln(x_patches + attn_out)
        
        # 4. MLP & Residual
        x_patches = x_patches + self.mlp(x_patches)
        
        # 5. Calculate focal weights (mean of attention directed TO each token across all heads)
        # attn_weights_patch shape is [num_patches, num_heads, patch_size, patch_size]
        # Sum over query dimension (dim=2) to see how much attention each key (dim=3) received
        # Average over heads (dim=1)
        focal_weights_patch = attn_weights_patch.sum(dim=2).mean(dim=1) # [num_patches, patch_size]
        focal_weights = focal_weights_patch.flatten()[:num_points]
        
        # 6. Flatten and Remove Padding
        out = x_patches.view(-1, x.shape[1])[:num_points]
        return out, focal_weights


class PTv3OcclusionGnn(torch.nn.Module):
    def __init__(self, 
                 in_channels=4,      # x, y, z + frame_time
                 out_channels=1,     # valid/invalid mask
                 hidden_channels=64, # Latent dimension
                 num_super_nodes=128, 
                 k=20,
                 patch_size=16,
                 num_heads=4,
                 temperature=0.1,
                 use_cylindrical_encoding=True,
                 **kwargs):
        super().__init__()
        self.k = k
        self.m = num_super_nodes 
        self.use_cylindrical_encoding = use_cylindrical_encoding
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size

        # --- 0. POSITIONAL ENCODER ---
        if self.use_cylindrical_encoding:
            # Lift cylindrical coords (r, sin_theta, cos_theta, z) into high-dim space
            self.pos_encoder = nn.Sequential(
                nn.Linear(4, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            local_in = (in_channels + hidden_channels) * 2
        else:
            local_in = in_channels * 2

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

        # --- 2. PTv3 OCCLUSION STREAM (Global Context) ---
        self.ptv3_block = PTv3Block(hidden_channels, patch_size, num_heads, temperature)
        
        # Learnable Ray-Gate (Linear + Sigmoid on focal weights)
        self.ray_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

        # --- 3. INDIVIDUALIZED CONTEXT (Cross-Attention) ---
        # Position-Aware Attention: Queries (local) attend to the geometric map (super-nodes)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels, 
            num_heads=num_heads, 
            batch_first=True
        )

        # --- 4. CLASSIFICATION HEAD ---
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

        # Phase 0: Positional Embedding
        pos_cart = x[:, :3]
        if self.use_cylindrical_encoding:
            r = torch.norm(pos_cart[:, :2], dim=-1, keepdim=True)
            theta = torch.atan2(pos_cart[:, 1], pos_cart[:, 0]).unsqueeze(-1)
            pos_cyl = torch.cat([r, torch.sin(theta), torch.cos(theta), pos_cart[:, 2:3]], dim=-1)
            pos_emb = self.pos_encoder(pos_cyl) # [N, hidden]
        
        # Phase 1: Local Representation
        if self.use_cylindrical_encoding:
            h_local_input = torch.cat([x, pos_emb], dim=-1)
            h_local = self.local_conv(h_local_input, batch)
        else:
            h_local = self.local_conv(x, batch) 

        # Phase 2: Super-Node Extraction & Serialization
        indices = fps(pos_cart, batch, ratio=self.m / x.size(0))
        
        # Phase 3: Intra-Patch Attention on Serialized Super-Nodes
        if self.use_cylindrical_encoding:
            # Reorder geometric map by 'rays'
            pos_cyl_super = pos_cyl[indices]
            sort_indices = cylindrical_serialize(pos_cyl_super)
            inverse_indices = torch.argsort(sort_indices)
            h_super_raw = pos_emb[indices][sort_indices]
            r_coords_sorted = pos_cyl_super[sort_indices, 0:1] # extract r for sorting
            
            # Pass sorted structure into block
            h_super_context_sorted, focal_weights_sorted = self.ptv3_block(h_super_raw, r_coords_sorted)
            
            # Unsort to match original frame
            h_super_context_raw = h_super_context_sorted[inverse_indices]
            focal_weights = focal_weights_sorted[inverse_indices]
        else:
            h_super_raw = h_local[indices]
            # Dummy r_coords if encoding not used
            r_coords = torch.norm(pos_cart[indices, :2], dim=-1, keepdim=True)
            h_super_context_raw, focal_weights = self.ptv3_block(h_super_raw, r_coords)

        # --- Ray-Gate ---
        # Apply learnable sigmoid gate to focal weights
        # focal_weights is [N_super]
        gate = self.ray_gate(focal_weights.unsqueeze(-1)) # [N_super, 1]
        h_super_context = h_super_context_raw * gate

        # Phase 4: Individualized Cross-Attention
        if self.use_cylindrical_encoding:
            # Add positional embeddings to Query to help identify spatial relevance
            q = (h_local + pos_emb).unsqueeze(0)
        else:
            q = h_local.unsqueeze(0) # [1, N, hidden]
            
        k_v = h_super_context.unsqueeze(0) # [1, M, hidden]
        
        attn_output, attn_weights = self.cross_attn(
            q, k_v, k_v, 
            need_weights=return_intermediate
        )
        h_context = attn_output.squeeze(0) 

        # Phase 5: Final Decision
        fused = torch.cat([h_local, h_context], dim=1) # [N, hidden * 2]
        out = self.classifier(fused)
        
        if return_intermediate:
            return out, {
                "h_local": h_local,
                "indices": indices,
                "h_super_context": h_super_context,
                "h_context": h_context,
                "intra_patch_attn": focal_weights, # return focal weights (ungated) for viz
                "ray_gate": gate.squeeze(-1),
                "attn_weights": attn_weights.squeeze(0) if attn_weights is not None else None
            }
        return out
