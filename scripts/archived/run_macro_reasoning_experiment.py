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
from mmwave_model_integrator.torch_training.models.MacroReasoningGnn import MacroReasoningGnn


def parse_args():
    parser = argparse.ArgumentParser(description="Run Macro-Reasoning GNN Experiment")
    parser.add_argument("--config_label", type=str, default="IcaRAus_macro_reasoning_gnn",
                        help="The config file name (without .py) to use.")
    parser.add_argument("--dataset_path", type=str, default="/home/david/Downloads/IcaRAus_datasets/IcaRAus_ugv_gnn_50fh_wilk_cpsl_north_1st_occluded_no_rt_gt_no_rt_pts_no_gt_filter_0_25_eps_10_min_20_sub_train",
                        help="Path to the dataset directory.")
    parser.add_argument("--checkpoint_path", type=str, default="working_dir/IcaRAus_gnn_IcaRAus_ds/IcaRAus_MacroReasoningGnn.pth",
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

    print(f"--- Setting up Macro-Reasoning Experiment: {args.config_label} ---")
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../configs/IcaRAus_gnn/{args.config_label}.py"))
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
        
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

    # Model Parameters from config
    model_cfg = config.model
    model_type_name = model_cfg.pop('type')
    model = MacroReasoningGnn(**model_cfg)

    runner = GNNRunner(
        model=model,
        state_dict_path=args.checkpoint_path,
        cuda_device="cuda:0" if torch.cuda.is_available() else "cpu",
        edge_radius=10.0,
        enable_downsampling=False,
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

    # 2. Layer-wise Ablation Analysis
    print(f"Generating Detailed Macro-Reasoning Layer Analysis for index {args.analysis_idx}...")
    nodes = dataset.get_node_data(args.analysis_idx)
    labels = dataset.get_label_data(args.analysis_idx)
    nodes_encoded = input_encoder.encode(nodes)
    
    runner.model.eval()
    with torch.no_grad():
        x = torch.tensor(nodes_encoded, dtype=torch.float32).to(runner.device)
        # 1. Full Forward Pass
        out, intermediates = runner.model(x, return_intermediate=True)
        final_predictions = torch.sigmoid(out).cpu().numpy()

        # 2. Nested Ablation Readout (Using partial Super-Node Reasoning)
        sn_ints = intermediates["super_node_intermediates"]
        h_local = intermediates["h_local"]
        
        if runner.model.use_cylindrical_encoding:
            # Re-run the start of forward to get pos_emb accurately
            pos_cart = x[:, :3]
            r = torch.norm(pos_cart[:, :2], dim=-1, keepdim=True)
            theta = torch.atan2(pos_cart[:, 1], pos_cart[:, 0]).unsqueeze(-1)
            
            # Match MacroReasoningGnn scale factors
            r_scale_factor = 1.0
            theta_scale_factor = 1.0
            
            pos_cyl = torch.cat([r * r_scale_factor,
                                torch.sin(theta) * theta_scale_factor, 
                                torch.cos(theta) * theta_scale_factor, 
                                pos_cart[:, 2:3]], dim=-1)
                                
            pos_emb = runner.model.pos_encoder(pos_cyl)
            q_base = h_local + pos_emb
        else:
            q_base = h_local

        h_dim = runner.model.hidden_channels
        total_sn_layers = runner.model.super_node_layers
        super_context_dim = h_dim * total_sn_layers
        
        layer_predictions = []
        for i in range(total_sn_layers):
            sn_features_up_to_i = [sn_ints[f"layer_{j}_features"] for j in range(i + 1)]
            sn_fused_i = torch.cat(sn_features_up_to_i, dim=1) 
            
            padding_needed = h_dim * (total_sn_layers - (i + 1))
            if padding_needed > 0:
                padding = torch.zeros(sn_fused_i.size(0), padding_needed, device=runner.device)
                h_super_context_i = torch.cat([sn_fused_i, padding], dim=1)
            else:
                h_super_context_i = sn_fused_i
            
            q = runner.model.query_proj(q_base).unsqueeze(0)
            k_v = h_super_context_i.unsqueeze(0)
            attn_output, _ = runner.model.cross_attn(q, k_v, k_v)
            h_context_i = attn_output.squeeze(0)
            
            fused_i = torch.cat([h_local, h_context_i], dim=1)
            pred_i = torch.sigmoid(runner.model.classifier(fused_i))
            layer_predictions.append(pred_i.cpu().numpy())

    # --- FIGURE 1: Diagnostic ---
    diag_save_path = os.path.join(args.output_dir, f"{args.config_label}_diagnostic_idx{args.analysis_idx}.png")
    plotter.plot_deep_edge_conv_diagnostic(
        nodes=nodes,
        gt_labels=labels,
        predictions=final_predictions,
        save_path=diag_save_path,
        show=False
    )
    print(f"Diagnostic figure saved to {diag_save_path}")

    # --- FIGURE 2: Layer Analysis (Super-Node Skeleton) ---
    analysis_save_path = os.path.join(args.output_dir, f"{args.config_label}_supernode_layer_analysis_idx{args.analysis_idx}.png")
    plotter.plot_deep_edge_conv_layer_analysis(
        nodes=nodes, # Pass full nodes
        model_outputs=sn_ints,
        layer_predictions=layer_predictions, 
        super_node_indices=intermediates["indices"].cpu().numpy(), # Pass super-node indices
        save_path=analysis_save_path,
        show=False
    )
    print(f"Super-node layer analysis figure saved to {analysis_save_path}")

    print(f"\n--- Experiment Completed. Results saved to {args.output_dir} ---")


if __name__ == "__main__":
    main()
