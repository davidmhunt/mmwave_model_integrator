import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
# load_dotenv("../.env")
# DATASET_PATH=os.getenv("DATASET_DIRECTORY")
# # DATASET_PATH=os.path.join("/home/david/Downloads")
# MODEL_TRAINING_DATASET_PATH=os.getenv("MODEL_TRAINING_DATASET_PATH")
# GENERATED_DATASETS_PATH=os.getenv("GENERATED_DATASETS_PATH")
# from mmwave_model_integrator.torch_training.models.SequentialDynamicEdgeConv import SequentialDynamicEdgeConv
from mmwave_model_integrator.torch_training.models.TwoStreamSpatioTemporalGnn import TwoStreamSpatioTemporalGnn
from cpsl_datasets.gnn_node_ds import GnnNodeDS

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.plotting.plotter_gnn_pc_processing import PlotterGnnPCProcessing
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
# from mmwave_model_integrator.torch_training.models.SequentialDynamicEdgeConv import SequentialDynamicEdgeConv
from mmwave_model_integrator.torch_training.models.SAGEGnn import SageGNNClassifier
from mmwave_model_integrator.torch_training.models.CrossAttentionGnn import CrossAttentionGnn
from mmwave_model_integrator.plotting.movie_generator_gnn import MovieGeneratorGNN


#initialize the dataset
# config_label = "IcaRAus_gnn_two_stream"
# dataset_label = "IcaRAus_gnn_50fh"
# dataset_path = os.path.join(DATASET_PATH,"IcaRAus_datasets","{}_train".format(dataset_label))

# DATASET_PATH="/data/IcaRAus/generated_datasets"
DATASET_PATH = "/home/david/Downloads/IcaRAus_datasets"
num_frames_history = 50
#key {no}_occluded_{rt or olp}_gt_{rt or olp}_pts_{no}_gt_filter
config_label = "IcaRAus_ugv_gnn_{}fh_wilk_cpsl_north_1st_occluded_no_rt_gt_no_rt_pts_no_gt_filter_0_25_eps_10_min_20_sub".format(num_frames_history)
dataset_path = os.path.join(DATASET_PATH,"{}_train".format(config_label))


dataset = GnnNodeDS(
    dataset_path=dataset_path,
    node_folder="nodes",
    label_folder="labels",
)
print(dataset_path)

#initialize the encoder and decoder
input_encoder = _NodeEncoder()
ground_truth_encoder = _GTNodeEncoder()
plotter = PlotterGnnPCProcessing()

#testing the output
# model = TwoStreamSpatioTemporalGnn(
#     hidden_channels=28,
#     out_channels=1,
#     k=40,
#     dropout=0.1
# )

model = CrossAttentionGnn(
    in_channels=4,
    out_channels=1,
    hidden_channels=28,
    k=4,
    num_super_nodes=128,
    num_heads=4,
)

runner = GNNRunner(
    model= model,
    # state_dict_path="/home/david/Documents/odometry/submodules/mmwave_model_integrator/scripts/working_dir/IcaRAus_gnn_IcaRAus_ds/IcaRAus_TwoStreamSpatioTemporalGnn_IcaRAus_ds_50fh_k4.pth",
    # state_dict_path="/home/david/Downloads/IcaRAus_TwoStreamSpatioTemporalGnn_IcaRAus_ds_50fh_k4.pth",
    state_dict_path="/home/david/Documents/odometry/submodules/mmwave_model_integrator/scripts/working_dir/IcaRAus_gnn_IcaRAus_ds/IcaRAus_CrossAttentionGnn.pth",
    cuda_device="cpu",
    edge_radius=10.0, #unused for this model
    enable_downsampling=False, #true for previous models
    downsample_keep_ratio=0.20,
    downsample_min_points=300,
    use_sigmoid=True
)

MOVIE_TEMP_DIRECTORY="/home/david/Downloads/IcaRAus_gnn_cross_attention_gnn"
movie_generator = MovieGeneratorGNN(
    gnn_ds=dataset,
    plotter=plotter,
    input_encoder=input_encoder,
    runner=runner,
    decoder=None,
    ground_truth_encoder=ground_truth_encoder,
    temp_dir_path=os.path.join(MOVIE_TEMP_DIRECTORY,config_label)
)

#plot a sample from the training dataset
idx = 1000 #980 good for imaging #3000 good on 10fp_20fh_0_50_th_5mRng_0_2_res_train
nodes = dataset.get_node_data(idx)
labels = dataset.get_label_data(idx)
print(nodes.shape)

#generate plot, but save as .png instead of showing
plotter.plot_compilation(
    input_data=nodes,
    gt_data=labels,
    input_encoder=input_encoder,
    ground_truth_encoder=ground_truth_encoder,
    runner=runner,
    show=True
)