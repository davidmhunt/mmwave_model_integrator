import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
from mmwave_model_integrator.config import Config
import mmwave_model_integrator.torch_training.trainers as trainers

sys.path.append("../")
from cpsl_datasets.gnn_node_ds import GnnNodeDS

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.plotting.plotter_gnn_pc_processing import PlotterGnnPCProcessing
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.CrossAttentionGnn import CrossAttentionGnn


def parse_args():
    parser = argparse.ArgumentParser(description="Run CrossAttention GNN Experiment")
    parser.add_argument("--config_label", type=str, default="IcaRAus_gnn_cross_attention_gnn",
                        help="The config file name (without .py) to use for model/training setup.")
    parser.add_argument("--dataset_path", type=str, default="/home/david/Downloads/IcaRAus_datasets/IcaRAus_ugv_gnn_50fh_wilk_cpsl_north_1st_occluded_no_rt_gt_no_rt_pts_no_gt_filter_0_25_eps_10_min_20_sub_train",
                        help="Path to the dataset directory.")
    parser.add_argument("--checkpoint_path", type=str, default="working_dir/IcaRAus_gnn_IcaRAus_ds/IcaRAus_CrossAttentionGnn.pth",
                        help="Path to load/save the model checkpoint.")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and only run evaluation/plotting.")
    parser.add_argument("--num_instances", type=int, default=5,
                        help="Number of instances to plot in the compilation grid.")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save the resulting plots.")
    parser.add_argument("--analysis_idx", type=int, default=1000,
                        help="Dataset index to use for the forward-pass ablation analysis.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Setting up Experiment with Config: {args.config_label} ---")
    config_path = f"../configs/IcaRAus_gnn/{args.config_label}.py"
    config = Config(config_path)
    # config.print_config()

    if not args.test_only:
        print("\n--- Phase 1: Training ---")
        trainer_config = config.trainer
        # Update checkpoint path if needed, though trainer config might dictate it
        trainer_class = getattr(trainers, trainer_config.pop('type'))
        trainer = trainer_class(**trainer_config)
        trainer.train_model()
        print("Training completed.")
    else:
        print("\n--- Skipping Training (Test Only Mode) ---")

    print("\n--- Phase 2: Evaluation & Plotting ---")
    
    # Initialize Dataset
    print(f"Loading Dataset from: {args.dataset_path}")
    dataset = GnnNodeDS(
        dataset_path=args.dataset_path,
        node_folder="nodes",
        label_folder="labels",
    )

    # Initialize Encoders and Plotter
    input_encoder = _NodeEncoder()
    ground_truth_encoder = _GTNodeEncoder()
    plotter = PlotterGnnPCProcessing()

    # Initialize Model & Runner
    # We use parameters similar to the test script. 
    # In a fully robust script, these would be pulled dynamically from config.model
    model = CrossAttentionGnn(
        in_channels=4,
        out_channels=1,
        hidden_channels=28,
        k=4,
        num_super_nodes=128,
        num_heads=4,
    )

    runner = GNNRunner(
        model=model,
        state_dict_path=args.checkpoint_path,
        cuda_device="cpu", # Can parameterize this
        edge_radius=10.0,
        enable_downsampling=False,
        downsample_keep_ratio=0.20,
        downsample_min_points=300,
        use_sigmoid=True
    )

    # 1. Multi-instance Compilation Plot (e.g., 5x3)
    print(f"Generating Multi-Instance Compilation ({args.num_instances} instances)...")
    np.random.seed(42) # For reproducible random samples
    indices = np.random.choice(dataset.num_frames, size=args.num_instances, replace=False)
    
    instances_nodes = []
    instances_labels = []
    for idx in indices:
        instances_nodes.append(dataset.get_node_data(idx))
        instances_labels.append(dataset.get_label_data(idx))

    compilation_save_path = os.path.join(args.output_dir, f"compilation_{args.num_instances}x3.png")
    plotter.plot_multi_instance_compilation(
        instances_nodes=instances_nodes,
        instances_labels=instances_labels,
        input_encoder=input_encoder,
        ground_truth_encoder=ground_truth_encoder,
        runner=runner,
        save_path=compilation_save_path,
        show=False
    )

    # 2. Forward Pass Ablation Analysis
    print(f"Generating Forward Pass Analysis for index {args.analysis_idx}...")
    nodes = dataset.get_node_data(args.analysis_idx)
    labels = dataset.get_label_data(args.analysis_idx)
    
    nodes_encoded = input_encoder.encode(nodes)
    
    # Need to run model manually to get intermediates
    # GNNRunner usually just does prediction. We will extract the model block.
    runner.model.eval()
    with torch.no_grad():
        x = torch.tensor(nodes_encoded, dtype=torch.float32).to(runner.device)
        # Using return_intermediate=True which we added to CrossAttentionGnn
        out, intermediates = runner.model(x, return_intermediate=True)
    
    analysis_save_path = os.path.join(args.output_dir, f"forward_pass_analysis_idx{args.analysis_idx}.png")
    plotter.plot_forward_pass_analysis(
        nodes=nodes,
        model_outputs=intermediates,
        gt_labels=labels,
        save_path=analysis_save_path,
        show=False
    )

    print("\n--- Experiment Completed Successfully ---")


if __name__ == "__main__":
    main()
