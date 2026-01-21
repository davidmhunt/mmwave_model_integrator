import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from mmwave_model_integrator.config import Config
import mmwave_model_integrator.torch_training.trainers as trainers

# from dotenv import load_dotenv
# load_dotenv("../.env")
# DATASET_PATH=os.getenv("DATASET_DIRECTORY")
# DATASET_PATH=os.path.join("/data/radnav/radnav_model_datasets")
# MODEL_TRAINING_DATASET_PATH=os.getenv("MODEL_TRAINING_DATASET_PATH")
# GENERATED_DATASETS_PATH=os.getenv("GENERATED_DATASETS_PATH")


sys.path.append("../")
from cpsl_datasets.gnn_node_ds import GnnNodeDS

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.plotting.plotter_gnn_pc_processing import PlotterGnnPCProcessing

#initialize the dataset
# config_label = "IcaRAus_gnn_base"
# config_label = "IcaRAus_gnn_hierarchical_anchor_net_IcaRAus_ds.py"
config_label = "IcaRAus_gnn_two_stream_IcaRAus_ds_k_4"
# config_label = "IcaRAus_gnn_ruiyang_test_model_IcaRAus_ds"

#initialize the encoder and decoder
input_encoder = _NodeEncoder()
ground_truth_encoder = _GTNodeEncoder()
plotter = PlotterGnnPCProcessing()


config_path = "../configs/IcaRAus_gnn/{}.py".format(config_label)
config = Config(config_path)

config.print_config()

trainer_config = config.trainer
trainer_class = getattr(trainers,trainer_config.pop('type'))
trainer = trainer_class(**trainer_config)

trainer.train_model()