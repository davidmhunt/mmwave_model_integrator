import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import csv

from mmwave_model_integrator.config import Config
import mmwave_model_integrator.torch_training.trainers as trainers

sys.path.append("../")
from cpsl_datasets.gnn_node_ds import GnnNodeDS

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.plotting.plotter_gnn_pc_processing import PlotterGnnPCProcessing
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.DensifyingDeepDynamicEdgeConvGnn import DensifyingDeepDynamicEdgeConvGnn
from torch_scatter import scatter_add

def parse_args():
    parser = argparse.ArgumentParser(description="Run Densifying Deep DynamicEdgeConv GNN Experiment")
    parser.add_argument("--config_label", type=str, default="IcaRAus_gnn_final_IcaRAus_ds",
                        help="The config file name (without .py) to use.")
    parser.add_argument("--dataset_path", type=str, default="/home/david/Downloads/IcaRAus_datasets/RaGNNarok_ugv_IcaRAus_ds_wilk_cpsl_north",
                        help="Path to the dataset directory.")
    parser.add_argument("--checkpoint_path", type=str, default="/home/david/Documents/odometry/submodules/mmwave_model_integrator/scripts/working_dir/IcaRAus_gnn_IcaRAus_ds/IcaRAus_gnn_IcaRAus_ds.pth",
                        help="Path to load/save the model checkpoint.")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and only run evaluation/plotting.")
    parser.add_argument("--num_instances", type=int, default=5,
                        help="Number of instances to plot in the compilation grid.")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save the resulting plots.")
    parser.add_argument("--analysis_idx", type=int, default=1000,
                        help="Dataset index to use for the layer-wise ablation analysis.")
    parser.add_argument("--print_stats", action="store_true",
                        help="Print prediction statistics during inference.")
    parser.add_argument("--skip_benchmark", action="store_true",
                        help="Skip performance benchmarking phase.")
    return parser.parse_args()

def benchmark_model(runner, dataset, input_encoder, device_name, config_label, output_dir, num_threads=None, num_warmup=10, num_runs=1000):
    """Measures average inference time on a specific device and breaks it down by component."""
    
    # Save original thread count and set new one if requested
    original_threads = torch.get_num_threads()
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    
    current_threads = torch.get_num_threads()
    mode_str = "Single-Core" if current_threads == 1 else "Multi-Core"
    print(f"\nRunning detailed benchmark on {device_name} ({mode_str}, Threads: {current_threads})...")
    
    # Move model to target device
    device = torch.device(device_name)
    runner.model.to(device)
    runner.device = device
    runner.model.eval()

    total_times = []
    component_times = {
        "subsampling": [],
        "backbone": [],
        "interpolation": [],
        "refinement": []
    }

    # Ensure model is on the correct device

    with torch.no_grad():
        # Prepare test data (use a consistent frame)
        nodes = dataset.get_node_data(0)
        nodes_encoded = input_encoder.encode(nodes)
        x = torch.tensor(nodes_encoded, dtype=torch.float32).to(device)

        # Warmup
        for _ in range(num_warmup):
            _ = runner.model(x, return_intermediate=True)
        
        if "cuda" in device_name:
            torch.cuda.synchronize()

        # Measurement
        import time
        for idx in range(num_runs):
            # Select frame (cycling through dataset if needed)
            data_idx = idx % dataset.num_frames
            nodes = dataset.get_node_data(data_idx)
            nodes_encoded = input_encoder.encode(nodes)
            x = torch.tensor(nodes_encoded, dtype=torch.float32).to(device)
            
            if "cuda" in device_name:
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _, intermediates = runner.model(x, return_intermediate=True)
            if "cuda" in device_name:
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            total_times.append((end - start) * 1000.0) # ms
            
            timing = intermediates.get("timing_ms", {})
            for key in component_times:
                if key in timing:
                    component_times[key].append(timing[key])

    avg_total = np.mean(total_times)
    std_total = np.std(total_times)
    
    print(f"\n--- Timing Breakdown ({device_name}, {current_threads} Threads) ---")
    print(f"{'Component':<20} | {'Mean (ms)':<10} | {'Std (ms)':<10} | {'Percentage':<10}")
    print("-" * 58)
    
    csv_rows = []
    for comp, times in component_times.items():
        if times:
            avg_comp = np.mean(times)
            std_comp = np.std(times)
            percentage = (avg_comp / avg_total) * 100.0
            print(f"{comp.capitalize():<20} | {avg_comp:10.2f} | {std_comp:10.2f} | {percentage:9.1f}%")
            csv_rows.append({
                "Component": comp.capitalize(),
                "Mean_ms": f"{avg_comp:.4f}",
                "Std_ms": f"{std_comp:.4f}",
                "Percentage": f"{percentage:.2f}%"
            })
    
    print("-" * 58)
    print(f"{'Total (Measured)':<20} | {avg_total:10.2f} | {std_total:10.2f} | 100.0%")
    print(f"Throughput: {1000.0 / avg_total:.2f} Hz")

    # CSV Export
    thread_suffix = f"_threads_{current_threads}"
    csv_path = os.path.join(output_dir, f"timing_breakdown_{config_label}_{device_name.replace(':', '_')}{thread_suffix}.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Component", "Mean_ms", "Std_ms", "Percentage"])
        writer.writeheader()
        writer.writerows(csv_rows)
        writer.writerow({
            "Component": "Total",
            "Mean_ms": f"{avg_total:.4f}",
            "Std_ms": f"{std_total:.4f}",
            "Percentage": "100.00%"
        })
    
    print(f"Detailed timing results saved to {csv_path}")

    # Restore original thread count
    torch.set_num_threads(original_threads)
    
    return avg_total, 1000.0 / avg_total

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Setting up Densifying Deep DynamicEdgeConv Experiment: {args.config_label} ---")
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
    
    dataset = GnnNodeDS(
        dataset_path=config.generated_dataset["generated_dataset_path"],
        node_folder="nodes",
        label_folder="labels",
    )

    input_encoder = _NodeEncoder()
    ground_truth_encoder = _GTNodeEncoder()
    plotter = PlotterGnnPCProcessing()

    config = Config(config_path)
    model_cfg = config.model
    model_type = model_cfg.pop('type')
    model = DensifyingDeepDynamicEdgeConvGnn(**model_cfg)

    dataset_cfg = config.trainer["dataset"]
    enable_downsampling = dataset_cfg.get("enable_downsampling", False)
    downsample_keep_ratio = dataset_cfg.get("downsample_keep_ratio", 1.0)
    downsample_min_points = dataset_cfg.get("downsample_min_points", 0)
    
    # Check for checkpoint existence in test-only mode
    if args.test_only and not os.path.exists(args.checkpoint_path):
        print(f"\n[WARNING] Checkpoint not found at {args.checkpoint_path}")
        print("Proceeding with Randomly Initialized weights for demonstration purposes.")
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint_path)
        print(f"Temporary dummy checkpoint saved to {args.checkpoint_path}")

    runner = GNNRunner(
        model=model,
        state_dict_path=args.checkpoint_path,
        cuda_device="cuda:0" if torch.cuda.is_available() else "cpu",
        edge_radius=10.0,
        enable_downsampling=enable_downsampling,
        downsample_keep_ratio=downsample_keep_ratio,
        downsample_min_points=downsample_min_points,
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
    print(f"Compilation saved to {compilation_save_path}")

    # 2. Detailed Layer-wise Analysis
    print(f"Generating Detailed Densifying Layer-wise Analysis for index {args.analysis_idx}...")
    nodes = dataset.get_node_data(args.analysis_idx)
    labels = dataset.get_label_data(args.analysis_idx)
    
    if runner.enable_downsampling:
        num_points = nodes.shape[0]
        target_count = int(num_points * runner.downsample_keep_ratio)
        target_count = max(target_count, runner.downsample_min_points)
        final_count = min(num_points, target_count)
        
        if final_count < num_points:
            np.random.seed(42)
            indices = np.random.permutation(num_points)[:final_count]
            nodes = nodes[indices]
            labels = labels[indices]

    nodes_encoded = input_encoder.encode(nodes)
    
    runner.model.eval()
    with torch.no_grad():
        x = torch.tensor(nodes_encoded, dtype=torch.float32).to(runner.device)
        out, intermediates = runner.model(x, return_intermediate=True)
        final_predictions = torch.sigmoid(out).cpu().numpy()

        backbone_ints = intermediates["backbone"]
        idx_sparse = intermediates["idx_sparse"]
        x_sparse = intermediates["x_sparse"]
        assign_idx = intermediates["assign_idx"]
        
        dense_idx, sparse_idx = assign_idx[0], assign_idx[1]
        dist = torch.norm(x[dense_idx, :3] - x_sparse[sparse_idx, :3], dim=-1)
        weights = 1.0 / (dist.pow(runner.model.p) + 1e-10)
        total_weight = scatter_add(weights, dense_idx, dim=0, dim_size=x.size(0))
        normalized_weights = weights / total_weight[dense_idx]
        
        # We need to manually densify each layer's accumulated super-node context
        h_dim = runner.model.backbone.hidden_channels
        total_layers = runner.model.backbone.num_layers
        
        layer_predictions = []
        for i in range(total_layers):
            features_up_to_i = [backbone_ints[f"layer_{j}_features"] for j in range(i + 1)]
            fused_sparse_i = torch.cat(features_up_to_i, dim=1) # [N_sparse, h_dim * (i+1)]
            
            padding_needed = h_dim * (total_layers - (i + 1))
            if padding_needed > 0:
                padding = torch.zeros(fused_sparse_i.size(0), padding_needed, device=runner.device)
                fused_sparse_padded = torch.cat([fused_sparse_i, padding], dim=1)
            else:
                fused_sparse_padded = fused_sparse_i
                
            # Interpolate to dense points!
            weighted_features = fused_sparse_padded[sparse_idx] * normalized_weights.unsqueeze(-1)
            fused_dense_i = scatter_add(weighted_features, dense_idx, dim=0, dim_size=x.size(0))
            
            x_norm = runner.model.input_norm(x)
            final_input = torch.cat([x_norm, fused_dense_i], dim=-1)
            
            pred_i = torch.sigmoid(runner.model.refiner(final_input))
            layer_predictions.append(pred_i.cpu().numpy())

    nodes_sparse = nodes[idx_sparse.cpu().numpy()]
    assign_idx_np = assign_idx.cpu().numpy()

    # --- FIGURE 1: Diagnostic ---
    diag_save_path = os.path.join(args.output_dir, f"{args.config_label}_diagnostic_idx{args.analysis_idx}.png")
    
    # Extract backbone output for coloring stars
    out_sparse = intermediates.get("out_sparse") # Might be missing if old model
    
    plotter.plot_densifying_diagnostic(
        nodes_dense=nodes,
        nodes_sparse=nodes_sparse,
        assign_idx=assign_idx_np,
        gt_labels=labels,
        predictions=final_predictions,
        sparse_predictions=out_sparse,
        save_path=diag_save_path,
        show=False
    )
    print(f"Diagnostic figure saved to {diag_save_path}")

    # --- FIGURE 2: Layer Analysis ---
    analysis_save_path = os.path.join(args.output_dir, f"{args.config_label}_layer_analysis_idx{args.analysis_idx}.png")
    plotter.plot_densifying_layer_analysis(
        nodes_dense=nodes,
        nodes_sparse=nodes_sparse,
        model_outputs=backbone_ints,
        layer_predictions=layer_predictions,
        save_path=analysis_save_path,
        show=False
    )
    print(f"Layer analysis figure saved to {analysis_save_path}")

    # --- PHASE 3: Benchmarking ---
    if not args.skip_benchmark:
        print("\n--- Phase 3: Performance Benchmarking ---")
        
        # 1. GPU Benchmark (Multi-core dispatch)
        if torch.cuda.is_available():
            benchmark_model(runner, dataset, input_encoder, "cuda:0", args.config_label, args.output_dir)
        
        # 2. CPU Multi-core Benchmark (System Default)
        benchmark_model(runner, dataset, input_encoder, "cpu", args.config_label, args.output_dir)

        # 3. CPU Single-core Benchmark
        benchmark_model(runner, dataset, input_encoder, "cpu", args.config_label, args.output_dir, num_threads=1)

    print(f"\n--- Experiment Completed. Results saved to {args.output_dir} ---")

if __name__ == "__main__":
    main()
