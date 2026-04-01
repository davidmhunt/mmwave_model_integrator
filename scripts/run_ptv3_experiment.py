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
from mmwave_model_integrator.torch_training.models.PTv3OcclusionGnn import PTv3OcclusionGnn


def parse_args():
    parser = argparse.ArgumentParser(description="Run PTv3 Occlusion GNN Experiment")
    parser.add_argument("--config_label", type=str, default="IcaRAus_ptv3_gnn",
                        help="The config file name (without .py) to use.")
    parser.add_argument("--dataset_path", type=str, default="/home/david/Downloads/IcaRAus_datasets/IcaRAus_ugv_gnn_50fh_wilk_cpsl_north_1st_occluded_no_rt_gt_no_rt_pts_no_gt_filter_0_25_eps_10_min_20_sub_train",
                        help="Path to the dataset directory.")
    parser.add_argument("--checkpoint_path", type=str, default="working_dir/IcaRAus_gnn_IcaRAus_ds/IcaRAus_PTv3OcclusionGnn.pth",
                        help="Path to load/save the model checkpoint.")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and only run evaluation/plotting.")
    parser.add_argument("--num_instances", type=int, default=5,
                        help="Number of instances to plot in the compilation grid.")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save the resulting plots.")
    parser.add_argument("--analysis_idx", type=int, default=1000,
                        help="Dataset index to use for the forward-pass ablation analysis.")
    parser.add_argument("--print_stats", action="store_true",
                        help="Print prediction statistics (min, max, mean) during inference.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Setting up PTv3 Occlusion Experiment: {args.config_label} ---")
    config_path = f"../configs/IcaRAus_gnn/{args.config_label}.py"
    config = Config(config_path)

    if not args.test_only:
        print("\n--- Phase 1: Training ---")
        trainer_config = config.trainer
        trainer_class = getattr(trainers, trainer_config.pop('type'))
        trainer = trainer_class(**trainer_config)
        trainer.train_model()
        print("Training completed.")
    else:
        print("\n--- Skipping Training (Test Only Mode) ---")

    print("\n--- Phase 2: Evaluation & Plotting ---")
    
    # Initialize Dataset
    dataset = GnnNodeDS(
        dataset_path=args.dataset_path,
        node_folder="nodes",
        label_folder="labels",
    )

    input_encoder = _NodeEncoder()
    ground_truth_encoder = _GTNodeEncoder()
    plotter = PlotterGnnPCProcessing()

    # Model Parameters (matching config for inference)
    model = PTv3OcclusionGnn(
        in_channels=4,
        out_channels=1,
        hidden_channels=28,
        k=4,
        num_super_nodes=128,
        patch_size=16,
        num_heads=4,
        use_cylindrical_encoding=True,
    )

    runner = GNNRunner(
        model=model,
        state_dict_path=args.checkpoint_path,
        cuda_device="cpu",
        edge_radius=10.0,
        enable_downsampling=False,
        downsample_keep_ratio=0.20,
        downsample_min_points=300,
        use_sigmoid=True,
        print_stats=args.print_stats
    )

    # 1. Multi-instance Compilation Plot
    print(f"Generating Multi-Instance Compilation...")
    np.random.seed(42)
    indices = np.random.choice(dataset.num_frames, size=args.num_instances, replace=False)
    
    instances_nodes = []
    instances_labels = []
    for idx in indices:
        instances_nodes.append(dataset.get_node_data(idx))
        instances_labels.append(dataset.get_label_data(idx))

    # UNIQUE NAME: prefix with config_label
    compilation_save_path = os.path.join(args.output_dir, f"{args.config_label}_compilation_{args.num_instances}x3.png")
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
    print(f"Generating PTv3 Occlusion Forward Pass Analysis for index {args.analysis_idx}...")
    nodes = dataset.get_node_data(args.analysis_idx)
    labels = dataset.get_label_data(args.analysis_idx)
    nodes_encoded = input_encoder.encode(nodes)
    
    runner.model.eval()
    with torch.no_grad():
        x = torch.tensor(nodes_encoded, dtype=torch.float32).to(runner.device)
        out, intermediates = runner.model(x, return_intermediate=True)
    
    # UNIQUE NAME: prefix with config_label
    analysis_save_path = os.path.join(args.output_dir, f"{args.config_label}_analysis_idx{args.analysis_idx}.png")
    plotter.plot_forward_pass_analysis(
        nodes=nodes,
        model_outputs=intermediates,
        gt_labels=labels,
        save_path=analysis_save_path,
        show=False
    )

    print(f"\n--- Experiment Completed. Results saved to {args.output_dir} ---")


if __name__ == "__main__":
    main()
