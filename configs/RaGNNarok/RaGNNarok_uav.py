_base_ = ['../sage_gnn_base.py']

config_label = "Hermes_60fh_0_1th" 

generated_dataset = dict(
    generated_dataset_path="/data/radnav/radnav_model_datasets/{}_train".format(config_label)
)

trainer = dict(
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "{}".format(config_label),
    dataset = dict(
        edge_radius = 1.0
    ),
    working_dir="working_dir/RaGNNarok",
    epochs=13
)
