_base_ = ["hermes_base.py"]
input_size = (64,128) #closest size to 100x90

model = dict(
    encoder_input_channels=1,
    input_dimmensions=input_size
    )

generated_dataset = dict(
    generated_dataset_path="/home/david/Downloads/generated_datasets/Hermes_train"
)

trainer = dict(
    model=model,
    dataset = dict(
        type='_BaseTorchDataset',
        input_transforms = [
            dict(type='ToTensor'),
            dict(
                type='Resize',
                size=input_size
            )
        ],
        output_transforms=[
            dict(type='ToTensor'),
            dict(
                type='Resize',
                size=input_size
            )
        ]
    ),
    save_name = "hermes_input_64_128",
)

