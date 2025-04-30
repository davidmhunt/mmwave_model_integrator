model = dict(
    type='ResNet18Nano',
    n_channels= 1,
    n_outputs=2
)

generated_dataset = dict(
    input_encoding_folder="doppler_az_resps",
    ground_truth_encoding_folder="vels",
    generated_dataset_path="/home/david/Downloads/generated_datasets/RadVel_train"
)

trainer = dict(
    type='_BaseTorchTrainer',
    model = model,
    optimizer = dict(
        type='Adam',
        lr=0.001
    ),
    dataset = dict(
        type='DopAzToVelDataset',
        input_transforms = [
            dict(
                type='Resize',
                size=(64,64)
            )
        ],
        output_transforms=[]
    ),
    data_loader = dict(
        type='DataLoader',
        batch_size=64,
        shuffle=True,
        num_workers=4
    ),
    dataset_path = generated_dataset["generated_dataset_path"],
    input_directory=generated_dataset["input_encoding_folder"],
    output_directory=generated_dataset["ground_truth_encoding_folder"],
    val_split = 0.15,
    working_dir = "working_dir",
    save_name = "RadVel",
    loss_fn = dict(
        type='RMSELoss'
    ),
    epochs = 15,
    pretrained_state_dict_path=None,
    cuda_device="cuda:0",
    multiple_GPUs=False
)