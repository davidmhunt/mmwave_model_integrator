_base_ = ["radcloud.py"]
input_size = (120,240) #closest size to 100x90

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
        ],
        paired_transforms=[
            dict(
                type='RandomRotationPair',
                degrees=15
            ),
            dict(
                type='RandomHorizontalFlipPair',
                p=0.5
            ),
            dict(
                type='RandomVerticalFlipPair',
                p=0.5
            ),
            # dict(
            #     type='RandomResizedCropPair',
            #     size=input_size,
            #     scale=(0.5,1.0),
            #     ratio=(0.75,1.33)
            # )
        ]
    ),
    data_loader = dict(
        batch_size=8
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    save_name = "Hermes_base",
    epochs=25
)

