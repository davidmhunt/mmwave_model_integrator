import torch

_base_ = ["IcaRAus_gnn_base.py"]
model = dict(
    type='TwoStreamSpatioTemporalGnn',
    hidden_channels=28,
    out_channels=1,
    k=40,
    dropout=0.1,
    in_channels=None
)
config_label = "IcaRAus_TwoStreamSpatioTemporalGnn_IcaRAus_ds_50fh"
dataset_label = "IcaRAus_gnn_50fh"

generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/IcaRAus_datasets/{}_train".format(dataset_label)
)

trainer = dict(
    model = model,
    dataset = dict(
        downsample_keep_ratio=0.20
    ),
    loss_fn = dict(
        pos_weight = torch.tensor([0.40])
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "{}".format(config_label),
    cuda_device="cuda:0"
)