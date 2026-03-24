_base_ = ["../radcloud.py"]
input_size = (160,160) #closest size to 100x90

model = dict(
    encoder_input_channels=1,
    input_dimmensions=input_size,
    encoder_out_channels= (64,128),
    decoder_input_channels= (256,128),
    )

generated_dataset = dict(
    generated_dataset_path="/data/IcaRAus/generated_datasets/IcaRAus_ugv_unet_50fh_wilk_cpsl_north_1st_no_occluded_rt_gt_olp_pts_gt_filter_train",
    input_encoding_folder="grids",
    ground_truth_encoding_folder="gt_grids"
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
        batch_size=32
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    input_directory=generated_dataset["input_encoding_folder"],
    output_directory=generated_dataset["ground_truth_encoding_folder"],
    save_name = "IcaRAus_unet_base",
    working_dir = "working_dir/IcaRAus_unet",
    epochs=25
)

