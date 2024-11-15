import sys
import os
import numpy as np

from dotenv import load_dotenv
load_dotenv()
CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")
RADCLOUD_MODEL_STATE_DICT_PATH=os.getenv("RADCLOUD_MODEL_STATE_DICT_PATH")

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_model_integrator.input_encoders.radcloud_encoder import RadCloudEncoder
from mmwave_model_integrator.model_runner.radcloud_runner import RadCloudRunner
from mmwave_model_integrator.decoders.radcloud_decoder import RadCloudDecoder
from mmwave_model_integrator.transforms.coordinate_transforms import polar_to_cartesian
from mmwave_model_integrator.real_time_runner.real_time_rng_az_to_pc_runner import realTimeRngAzToPCRunner

#load the configuration
cfg_manager = ConfigManager()
cfg_path = os.path.join(CONFIG_DIRECTORY,"RadCloud.cfg")
cfg_manager.load_cfg(cfg_path)
cfg_manager.compute_radar_perforance(profile_idx=0)

#load the input encoder
input_encoder = RadCloudEncoder(
    config_manager=cfg_manager,
    max_range_bin=64,
    num_chirps_to_encode=40,
    radar_fov_rad= [-0.87,0.87],
    num_az_angle_bins=64,
    power_range_dB=[60,105]
)

#load the runner
runner = RadCloudRunner(
    state_dict_path=RADCLOUD_MODEL_STATE_DICT_PATH,
    cuda_device="cpu"
)

#load the prediction decoder
prediction_decoder = RadCloudDecoder(
    max_range_m=8.56,
    num_range_bins=64,
    angle_range_rad=[np.deg2rad(50),np.deg2rad(-50)],#[-np.pi/2 - 0.87,-np.pi/2 + 0.87],
    num_angle_bins=48
)

#initialize the real-time runner
real_time_runner = realTimeRngAzToPCRunner(
    input_encoder=input_encoder,
    model_runner=runner,
    prediction_decoder=prediction_decoder,
    adc_cube_addr=6001,
    input_encoding_addr=6003,#6003,
    predicted_pc_addr=6004,#6004
)

#run the real-time runner
real_time_runner.run()