_base_ = ["sage_gnn_base.py"]

model = dict(
    in_channels=6
)

config_label = "10fp_80fh_0_50_th_5mRng_0_2_res"

generated_dataset = dict(
    generated_dataset_path="/data/radnav/radnav_model_datasets/{}_train".format(config_label)
)

trainer = dict(
    model=model,
    dataset = dict(
        enable_cylindrical_encoding=True
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "Sage_{}_cylindrical".format(config_label)
)
