import torch

_base_ = ["IcaRAus_gnn_base.py"]
model = dict(
    type='TwoStreamSpatioTemporalGnn',
    hidden_channels=32,
    out_channels=1,
    k=10,
    dropout=0.5,
    in_channels=None
)
config_label = "IcaRAus_TwoStreamSpatioTemporalGnn_100fh"

trainer = dict(
    model = dict(
        type='TwoStreamSpatioTemporalGnn',
        hidden_channels=32,
        out_channels=1,
        k=10
    ),
    save_name = "{}".format(config_label)
)