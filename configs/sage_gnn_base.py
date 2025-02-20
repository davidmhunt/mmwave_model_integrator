model = dict(
    type='SageGNNClassifier',
    in_channels = 4,
    hidden_channels=16,
    out_channels=1
)
config_label = "10fp_80fh_0_50_th_5mRng_0_2_res"

generated_dataset = dict(
    input_encoding_folder="nodes",
    ground_truth_encoding_folder="labels",
    generated_dataset_path="/data/radnav/radnav_model_datasets/{}_train".format(config_label)
)

trainer = dict(
    type='GNNTorchTrainer',
    model = model,
    optimizer = dict(
        type='Adam',
        lr=0.001
    ),

    loss_fn = dict(
        type='BCELoss'
    ),
    dataset = dict(
        type='_GnnNodeDataset',
        edge_radius=10.0,
        transforms=[],
        enable_random_yaw_rotate=True,
        enable_occupancy_grid_preturbations=True,
        enable_x_y_position_preturbations=True,
    ),
    data_loader = dict(
        type='TGDataLoader',
        batch_size=256,
        shuffle=True,
        num_workers=8
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    node_directory=generated_dataset["input_encoding_folder"],
    label_directory=generated_dataset["ground_truth_encoding_folder"],
    val_split = 0.25,
    working_dir = "working_dir",
    save_name = "Sage_{}".format(config_label),
    epochs = 10,
    pretrained_state_dict_path=None,
    cuda_device="cuda:0",
    multiple_GPUs=False
)