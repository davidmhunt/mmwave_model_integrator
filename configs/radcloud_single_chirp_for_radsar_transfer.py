_base_ = ["radcloud.py"]

model = dict(
    encoder_input_channels=1,
    input_dimmensions=(96,88) #closest size to 100x90
    )

generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/generated_datasets/RadCloud_train"
)

trainer = dict(
    model=model,
    dataset = dict(
        type='_BaseTorchDataset',
        input_transforms = [
            dict(type='ToTensor'),
            dict(
                type='Resize',
                size=(96,88)
            )
        ],
        output_transforms=[
            dict(type='ToTensor'),
            dict(
                type='Resize',
                size=(96,88)
            )
        ]
    ),
    data_loader = dict(
        batch_size=128,
        num_workers=10
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "RadCloud_1_chirp_10e_radsar_transfer"
)

