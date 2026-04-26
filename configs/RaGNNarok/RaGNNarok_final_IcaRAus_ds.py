_base_ = ['../sage_gnn_base.py']

config_label = "RaGNNarok_final_IcaRAus_ds" 

dataset_label = "RaGNNarok_ugv_IcaRAus_ds_wilk_cpsl_north"
generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/IcaRAus_datasets/{}_train".format(dataset_label)
)

trainer = dict(
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "{}".format(config_label),
    dataset = dict(
        edge_radius = 10.0
    ),
    working_dir="working_dir/RaGNNarok",
    epochs=50
)
