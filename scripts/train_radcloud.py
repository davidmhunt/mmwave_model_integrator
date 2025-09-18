import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()
DATASET_PATH=os.getenv("DATASET_DIRECTORY")
CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")
RADCLOUD_MODEL_STATE_DICT_PATH=os.getenv("RADCLOUD_MODEL_STATE_DICT_PATH")
MODEL_TRAINING_DATASET_PATH=os.getenv("MODEL_TRAINING_DATASET_PATH")
GENERATED_DATASETS_PATH=os.getenv("GENERATED_DATASETS_PATH")


sys.path.append("../")
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from cpsl_datasets.cpsl_ds import CpslDS

from mmwave_model_integrator.input_encoders.radcloud_encoder import RadCloudEncoder
from mmwave_model_integrator.ground_truth_encoders.radcloud_gt_encoder import RadCloudGTEncoder
from mmwave_model_integrator.plotting.plotter_rng_az_to_pc import PlotterRngAzToPC
from mmwave_model_integrator.dataset_generators.rng_az_to_pc_dataset_generator import RngAzToPCDatasetGenerator
import mmwave_model_integrator.torch_training.trainers as trainers


#setup the config manager
cfg_manager = ConfigManager()

cfg_path = os.path.join(CONFIG_DIRECTORY,"RadCloud_original.cfg")
cfg_manager.load_cfg(cfg_path)
cfg_manager.compute_radar_perforance(profile_idx=0)

#determine the paths to all of the datasets
dataset_groups = ["ugv_seen_dataset","ugv_unseen_dataset","ugv_rapid_movement_dataset"]

train_scenario_folders = []
test_scenario_folders = []

for group in dataset_groups:
    group_path = os.path.join(MODEL_TRAINING_DATASET_PATH,"RadCloud",group)
    entries = sorted(os.listdir(group_path))
    for entry in entries:
        path = os.path.join(group_path,entry)
        if os.path.isdir(path):
            if 'test' in entry.lower():
                test_scenario_folders.append(path)
            else:
                train_scenario_folders.append(path)

#initializing the dataset generator
dataset_path = train_scenario_folders[0]
dataset = CpslDS(
    dataset_path=dataset_path,
    radar_folder="radar",
    lidar_folder="lidar",
    camera_folder="camera",
    imu_orientation_folder="imu_data",
    imu_full_folder="imu_data_full"
)

#initialize the encoder and decoder
input_encoder = RadCloudEncoder(
    config_manager=cfg_manager,
    max_range_bin=64,
    num_chirps_to_encode=1,
    radar_fov_rad= [-0.87,0.87],
    num_az_angle_bins=64,
    power_range_dB=[60,105]
)

ground_truth_encoder = RadCloudGTEncoder(
    max_range_m=8.56,
    num_range_bins=64,
    angle_range_rad=[np.deg2rad(50),np.deg2rad(-50)],
    num_angle_bins=48,
    num_previous_frames=0
)

#initialize the dataset generator
generated_dataset_path = os.path.join(GENERATED_DATASETS_PATH,"RadCloud_train")
dataset_generator = RngAzToPCDatasetGenerator(
    generated_dataset_path=generated_dataset_path,
    dataset_handler=dataset,
    input_encoder=input_encoder,
    ground_truth_encoder=ground_truth_encoder,
    generated_file_name="frame",
    input_encoding_folder="x_s",
    ground_truth_encoding_folder="y_s",
    clear_existing_data=True
)

dataset_generator.generate_dataset_from_multiple_scenarios(train_scenario_folders)

from mmwave_model_integrator.config import Config
# config_path = "../configs/radcloud_single_chirp.py"
config_path = "configs/radcloud_single_chirp_for_radsar_transfer.py"
config = Config(config_path)

config.print_config()

#start training
trainer_config = config.trainer
trainer_class = getattr(trainers,trainer_config.pop('type'))
trainer = trainer_class(**trainer_config)

#train the model
trainer.train_model()