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
    global_dim=32,
    use_fourier_features=True
)
config_label = "IcaRAus_Ruiyang_test_model_IcaRAus_ds_gnn_global_focal"
dataset_label = "IcaRAus_gnn_50fh"

generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/IcaRAus_datasets/{}_train".format(dataset_label)
)

loss_fn = dict(
    type='FocalLoss',
    alpha=0.10,
    gamma=3.0
)

# loss_fn = dict(
#     pos_weight = torch.tensor([0.40])
# )

trainer = dict(
    model = model,
    dataset = dict(
        enable_downsampling=False,
        downsample_keep_ratio=0.20
    ),
    loss_fn = loss_fn,
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "{}".format(config_label),
    cuda_device="cuda:0"
)