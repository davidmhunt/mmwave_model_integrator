import torch

_base_ = ["IcaRAus_gnn_base.py"]

model = dict(
    type='DeepDynamicEdgeConvGnn',
    in_channels=4,
    out_channels=1,
    hidden_channels=32,
    num_layers=4,
    k=5,
    dropout=0.5
)

config_label = "IcaRAus_DeepDynamicEdgeConvGnn_k5_l4"

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
        downsample_keep_ratio=0.3,
        downsample_min_points=300
    ),
    epochs=30,
    cuda_device="cuda:0",
)
