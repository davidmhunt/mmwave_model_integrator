import torch

_base_ = ["IcaRAus_gnn_base.py"]
model = dict(
    type='RuiyangTestModel',
    hidden_channels=28,
    out_channels=1,
    k=4,
    dropout=0.1,
    in_channels=None,
    use_gnn=True,
    use_global_context=True,
    encoded_global_dim=3
)
config_label = "IcaRAus_Ruiyang_test_model_IcaRAus_ds_global_gnn"
dataset_label = "IcaRAus_gnn_50fh"

generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/IcaRAus_datasets/{}_train".format(dataset_label)
)

trainer = dict(
    model = model,
    dataset = dict(
        enable_downsampling=False,
        downsample_keep_ratio=0.20
    ),
    loss_fn = dict(
        pos_weight = torch.tensor([0.40])
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "{}".format(config_label),
    cuda_device="cuda:0"
)