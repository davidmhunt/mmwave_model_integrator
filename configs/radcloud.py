model = dict(
    type='RadCloudUnet',
    encoder_input_channels= 40,
    encoder_out_channels= (64,128,256),
    decoder_input_channels= (512,256,128),
    decoder_out_channels= 64,
    output_channels= 1,
    retain_dimmension= False,
    input_dimmensions= (64,48)
)

generated_dataset = dict(
    input_encoding_folder="x_s",
    ground_truth_encoding_folder="y_s",
    generated_dataset_path="/home/david/Downloads/generated_datasets/RadCloud_train"
)

trainer = dict(
    type='_BaseTorchTrainer',
    model = model,
    optimizer = dict(
        type='Adam',
        lr=0.001
    ),
    dataset = dict(
        type='_BaseTorchDataset',
        input_transforms = [
            dict(
                type='RandomRadarNoise',
                noise_level=0.0
            ),
            dict(type='ToTensor'),
            dict(
                type='Resize',
                size=(64,48)
            )
        ],
        output_transforms=[
            dict(type='ToTensor'),
            dict(
                type='Resize',
                size=(64,48)
            )
        ]
    ),
    data_loader = dict(
        type='DataLoader',
        batch_size=256,
        shuffle=True,
        num_workers=4
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    input_directory=generated_dataset["input_encoding_folder"],
    output_directory=generated_dataset["ground_truth_encoding_folder"],
    val_split = 0.15,
    working_dir = "working_dir",
    save_name = "RadCloud_40_chirps_10e",
    loss_fn = dict(
        type='BCE_DICE_Loss'
    ),
    epochs = 10,
    pretrained_state_dict_path=None,
    cuda_device="cuda:0",
    multiple_GPUs=False
)