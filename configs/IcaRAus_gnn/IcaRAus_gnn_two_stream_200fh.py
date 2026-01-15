import torch

_base_ = ["IcaRAus_gnn_two_stream.py"]

config_label = "IcaRAus_TwoStreamSpatioTemporalGnn_200fh"
dataset_label = "IcaRAus_gnn_200fh"

generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/{}_train".format(dataset_label)
)

trainer = dict(
    data_loader = dict(
        type='TGDataLoader',
        batch_size=40,
        shuffle=True,
        num_workers=18
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "{}".format(config_label),
    cuda_device="cuda:1"
)