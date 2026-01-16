import torch

_base_ = ["IcaRAus_gnn_two_stream.py"]

model = dict(
    type='SequentialDynamicEdgeConv',
    in_channels = 4,
    hidden_channels=32,
    out_channels=1,
    dropout=0.5,
    k=35
)

config_label = "IcaRAus_TwoStreamSpatioTemporalGnn_200fh"
dataset_label = "IcaRAus_gnn_200fh"

generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/{}_train".format(dataset_label)
)

trainer = dict(
    model = model,
    data_loader = dict(
        type='TGDataLoader',
        batch_size=40,
        shuffle=True,
        num_workers=18
    ),
    dataset = dict(
        downsample_keep_ratio=0.25
    ),
    loss_fn = dict(
        pos_weight = torch.tensor([0.25])
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "{}".format(config_label),
    cuda_device="cuda:1"
)