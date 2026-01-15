import torch

model = dict(
    type='SequentialDynamicEdgeConv',
    in_channels = 4,
    hidden_channels=32,
    out_channels=1,
    dropout=0.5,
    k=10
)
config_label = "IcaRAus_SequentialDynamicEdge_100fh"
dataset_label = "IcaRAus_gnn_100fh"
generated_dataset = dict(
    input_encoding_folder="nodes",
    ground_truth_encoding_folder="labels",
    generated_dataset_path="/data/radnav/radnav_model_datasets/{}_train".format(dataset_label)
)

trainer = dict(
    type='GNNTorchTrainer',
    model = model,
    optimizer = dict(
        type='Adam',
        lr=0.001
    ),

    loss_fn = dict(
        type='BCEWithLogitsLoss',
        # pos_weight=torch.tensor([20.0])
    ),
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
        enable_downsampling=True,
        downsample_keep_ratio=0.3,
        downsample_min_points=300
    ),
    data_loader = dict(
        type='TGDataLoader',
        batch_size=50,
        shuffle=True,
        num_workers=18
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    node_directory=generated_dataset["input_encoding_folder"],
    label_directory=generated_dataset["ground_truth_encoding_folder"],
    val_split = 0.25,
    working_dir = "working_dir/IcaRAus_gnn",
    save_name = "{}".format(config_label),
    epochs = 20,
    pretrained_state_dict_path=None,
    cuda_device="cuda:0",
    multiple_GPUs=False
)