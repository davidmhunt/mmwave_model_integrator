import torch

_base_ = ["IcaRAus_gnn_base.py"]

model = dict(
    type='DensifyingDeepDynamicEdgeConvGnn',
    in_channels=4,
    out_channels=1,
    hidden_channels=32,
    num_layers=4,
    k=4,
    p=2.0,
    num_sparse_points_fps=100,
    use_density_filtering=False,
    density_eps=0.1,
    density_min_samples=5,
    dropout=0.5
)

config_label = "IcaRAus_DensifyingDeepDynamicEdgeConvGnn_100_sparse_points_k4"

trainer = dict(
    model = model,
    optimizer = dict(
        type='Adam',
        lr=0.001,
        weight_decay=1e-5
    ),
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
        downsample_keep_ratio=0.1,
        downsample_min_points=200
    ),   
    epochs=100,
    cuda_device="cuda:1",
)
