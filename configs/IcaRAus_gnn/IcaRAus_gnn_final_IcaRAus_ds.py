import torch

_base_ = ["IcaRAus_gnn_base.py"]

model = dict(
    type='DensifyingDeepDynamicEdgeConvGnn',
    in_channels=5,
    out_channels=1,
    hidden_channels=32,
    num_layers=4,
    k=20,
    p=2.0,
    num_sparse_points_fps=100,
    use_density_filtering=False,
    density_eps=0.1,
    density_min_samples=5,
    dropout=0.5
)

dataset_label = "IcaRAus_ugv_IcaRAus_ds_wilk_cpsl_north_1st"
generated_dataset = dict(
    input_encoding_folder="nodes",
    ground_truth_encoding_folder="labels",
    generated_dataset_path="/home/david/Downloads/IcaRAus_datasets/{}_train".format(dataset_label)
)

config_label = "IcaRAus_gnn_IcaRAus_ds"
[]
trainer = dict(
    model = model,
    optimizer = dict(
        type='Adam',
        lr=0.001,
        weight_decay=1e-5
    ),
    loss_fn = dict(
        type='BCEWithLogitsLoss',
        pos_weight=torch.tensor([1.0])
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
    dataset_path = generated_dataset["generated_dataset_path"],
    node_directory=generated_dataset["input_encoding_folder"],
    label_directory=generated_dataset["ground_truth_encoding_folder"],
    epochs=100,
    cuda_device="cuda:1",
)
