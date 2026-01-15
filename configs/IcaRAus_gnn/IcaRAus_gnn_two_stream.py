import torch

_base_ = ["IcaRAus_gnn_base.py"]
model = dict(
    type='TwoStreamSpatioTemporalGnn',
    hidden_channels=32,
    out_channels=1,
    k=30,
    dropout=0.1,
    in_channels=None
)
config_label = "IcaRAus_TwoStreamSpatioTemporalGnn_100fh"

trainer = dict(
    model = model,
    save_name = "{}".format(config_label)
)