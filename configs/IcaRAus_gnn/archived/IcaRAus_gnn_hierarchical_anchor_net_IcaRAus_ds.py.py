import torch

_base_ = ["IcaRAus_gnn_base.py"]
model = dict(
    type='HierarchicalAnchorNet',
    in_channels=4,
    hidden_channels=64,
    out_channels=1,
    num_anchors=128,
    use_attention_aggregator=False
)
config_label = "IcaRAus_HierarchicalAnchorNet_IcaRAus_ds_50fh"
dataset_label = "IcaRAus_gnn_50fh"

generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/IcaRAus_datasets/{}_train".format(dataset_label)
)

trainer = dict(
    model = model,
    dataset = dict(
        enable_downsampling=False,
        downsample_keep_ratio=0.2
    ),
    loss_fn = dict(
        pos_weight = torch.tensor([0.40])
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "{}".format(config_label),
    cuda_device="cuda:0"
)