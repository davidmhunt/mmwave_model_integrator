import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from cpsl_datasets.gnn_node_ds import GnnNodeDS

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.plotting.plotter_gnn_pc_processing import PlotterGnnPCProcessing
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.SAGEGnn import SageGNNClassifier
from mmwave_model_integrator.plotting.movie_generator_gnn import MovieGeneratorGNN

sys.path.append("../")
from dotenv import load_dotenv
load_dotenv("../.env")
DATASET_PATH=os.getenv("DATASET_DIRECTORY")
MODEL_TRAINING_DATASET_PATH=os.getenv("MODEL_TRAINING_DATASET_PATH")
GENERATED_DATASETS_PATH=os.getenv("GENERATED_DATASETS_PATH")

config_label = "RaGNNarok_1fp_20fh_0_50_th_5mRng_0_2_res"
working_dir_path = "/home/david/Documents/odometry/submodules/mmwave_model_integrator/scripts/working_dir/RaGNNarok"
#initialize the dataset
dataset_path = os.path.join(DATASET_PATH,"{}_eval_imaging".format(config_label))
dataset = GnnNodeDS(
    dataset_path=dataset_path,
    node_folder="nodes",
    label_folder="labels"
)

#initialize the encoder and decoder
input_encoder = _NodeEncoder()
ground_truth_encoder = _GTNodeEncoder()
plotter = PlotterGnnPCProcessing()

#testing the output
runner = GNNRunner(
    model= SageGNNClassifier(
        in_channels=4,
        hidden_channels=16,
        out_channels=1
    ),state_dict_path=os.path.join(working_dir_path,"{}.pth".format(config_label)),
    cuda_device="cuda:0",
    edge_radius=10.0
)

movie_generator = MovieGeneratorGNN(
    gnn_ds=dataset,
    plotter=plotter,
    input_encoder=input_encoder,
    runner=runner,
    decoder=None,
    ground_truth_encoder=ground_truth_encoder,
    temp_dir_path=os.path.join(os.getenv("MOVIE_TEMP_DIRECTORY"),"RaGNNarok","{}_model_movie".format(config_label))
)

movie_generator.initialize_figure(
    nrows=1,
    ncols=3,
    figsize=(15,5),
    wspace=0.3,
    hspace=0.3
)

movie_generator.generate_movie_frames()
movie_generator.save_movie(video_file_name="{}.mp4".format(config_label),fps=20)