import torch

_base_ = ["IcaRAus_gnn_base.py"]
model = dict(
    type='CrossAttentionGnn',
    in_channels=4,
    out_channels=1,
    hidden_channels=28,
    k=4,
    num_super_nodes=128,
    num_heads=4,
)
config_label = "IcaRAus_CrossAttentionGnn"

trainer = dict(
    model = model,
    loss_fn = dict(
        type='BCEWithLogitsLoss',
        pos_weight=torch.tensor([0.25])
    ),
    save_name = "{}".format(config_label),
    dataset = dict(
        type='_GnnNodeDataset',
        edge_radius=10.0,
        transforms=[],
        enable_edge_attr=False,
        enable_edge_index=False,
        enable_random_yaw_rotate=True,
        enable_occupancy_grid_preturbations=False,
        enable_node_value_preturbations=True,
        node_value_preturbation_sigma=0.05,
        enable_x_y_position_preturbations=True,
        enable_cylindrical_encoding=False,
        enable_downsampling=False,
        downsample_keep_ratio=0.3,
        downsample_min_points=300
    ),
)