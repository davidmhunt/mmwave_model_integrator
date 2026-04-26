import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv("../.env")
# DATASET_PATH=os.getenv("DATASET_DIRECTORY")
DATASET_PATH=os.path.join("/home/david/Downloads/ragnnarok_datasets/")


sys.path.append("../")
from cpsl_datasets.gnn_node_ds import GnnNodeDS

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.plotting.plotter_gnn_pc_processing import PlotterGnnPCProcessing
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.SAGEGnn import SageGNNClassifier
from mmwave_model_integrator.plotting.movie_generator_gnn import MovieGeneratorGNN

#initialize the dataset
config_label = "RaGNNarok_final_RaGNNarok_ds"
# config_label = "RaGNNarok_final_IcaRAus_ds"

dataset_label = "RaGNNarok_ugv_RaGNNarok_ds_wilk_basement"
# dataset_label = "RaGNNarok_ugv_IcaRAus_ds_wilk_cpsl_north"
dataset_path = os.path.join(DATASET_PATH,"{}_train".format(dataset_label))
dataset = GnnNodeDS(
    dataset_path=dataset_path,
    node_folder="nodes",
    label_folder="labels"
)
print(dataset_path)

#initialize the encoder and decoder
input_encoder = _NodeEncoder()
ground_truth_encoder = _GTNodeEncoder()
plotter = PlotterGnnPCProcessing()

#testing the output

state_dict_path = "/home/david/Documents/odometry/submodules/mmwave_model_integrator/scripts/working_dir/RaGNNarok/RaGNNarok_final_RaGNNarok_ds.pth"

runner = GNNRunner(
    model= SageGNNClassifier(
        in_channels=4,
        hidden_channels=16,
        out_channels=1
    ),state_dict_path=state_dict_path,
    cuda_device="cpu",
    edge_radius=10.0,
    use_sigmoid=False
)

movie_generator = MovieGeneratorGNN(
    gnn_ds=dataset,
    plotter=plotter,
    input_encoder=input_encoder,
    runner=runner,
    decoder=None,
    ground_truth_encoder=ground_truth_encoder,
    temp_dir_path=os.path.join(os.getenv("MOVIE_TEMP_DIRECTORY"),config_label)
)

#plot a sample from the training dataset
idx = 100 #980 good for imaging #3000 good on 10fp_20fh_0_50_th_5mRng_0_2_res_train
nodes = dataset.get_node_data(idx)
labels = dataset.get_label_data(idx)
plotter.plot_compilation(
    input_data=nodes,
    gt_data=labels,
    input_encoder=input_encoder,
    ground_truth_encoder=ground_truth_encoder,
    runner=runner,
    show=True
)