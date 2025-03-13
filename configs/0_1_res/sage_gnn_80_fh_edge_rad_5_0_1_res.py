_base_ = ['../sage_gnn_base.py']

config_label = "10fp_80fh_0_50_th_5mRng_0_1_res" 

generated_dataset = dict(
    generated_dataset_path="/data/radnav/radnav_model_datasets/{}_train".format(config_label)
)

trainer = dict(
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "Sage_{}_edge_rad_5".format(config_label),
    dataset = dict(
        edge_radius = 5.0
    ),
    working_dir="working_dir/0_1_res",
    epochs=20
)
