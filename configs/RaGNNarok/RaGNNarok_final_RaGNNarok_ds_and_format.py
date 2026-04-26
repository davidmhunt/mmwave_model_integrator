_base_ = ['../sage_gnn_base.py']

config_label = "RaGNNarok_final_RaGNNarok_ds_and_format" 

model = dict(
    type='SageGNNClassifier',
    in_channels = 5,
    hidden_channels=16,
    out_channels=1
)

dataset_label = "IcaRAus_ugv_IcaRAus_ds_wilk_cpsl_north_1st"
generated_dataset = dict(
    input_encoding_folder="nodes",
    ground_truth_encoding_folder="labels",
    generated_dataset_path="/home/david/Downloads/IcaRAus_datasets/{}_train".format(dataset_label)
)

trainer = dict(
    model=model,
    dataset_path = generated_dataset["generated_dataset_path"],
    node_directory=generated_dataset["input_encoding_folder"],
    label_directory=generated_dataset["ground_truth_encoding_folder"],
    save_name = "{}".format(config_label),
    dataset = dict(
        edge_radius = 10.0
    ),
    working_dir="working_dir/RaGNNarok",
    epochs=50
)
