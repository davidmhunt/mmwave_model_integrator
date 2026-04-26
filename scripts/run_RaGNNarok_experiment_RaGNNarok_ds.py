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
from mmwave_model_integrator.torch_training.models.SAGEGnn import SageGNNClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Run SageGNNClassifier RaGNNarok Experiment")
    parser.add_argument("--config_label", type=str, default="RaGNNarok_final_RaGNNarok_ds",
                        help="The config file name (without .py) to use.")
    parser.add_argument("--checkpoint_path", type=str, default="/home/david/Documents/odometry/submodules/mmwave_model_integrator/scripts/working_dir/RaGNNarok/RaGNNarok_final_RaGNNarok_ds.pth",
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
    parser.add_argument("--num_runs", type=int, default=1000,
                        help="Number of runs for the benchmark.")
    return parser.parse_args()

def benchmark_model(runner, dataset, input_encoder, device_name, config_label, output_dir, num_threads=None, num_warmup=10, num_runs=1000):
    """Measures average inference time on a specific device."""
    
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

    graph_times = []
    model_times = []
    total_times = []

    with torch.no_grad():
        # Prepare test data (use a consistent frame, build edge index using radius graph)
        from torch_geometric.nn import radius_graph
        nodes = dataset.get_node_data(0)
        nodes_encoded = input_encoder.encode(nodes)
        x = torch.tensor(nodes_encoded, dtype=torch.float32).to(device)
        edge_index = radius_graph(x[:, :2], r=10.0, loop=False)

        # Warmup
        for _ in range(num_warmup):
            _ = runner.model(x, edge_index=edge_index)
        
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
            
            start_iter = time.perf_counter()
            
            # 1. Graph Generation
            start_graph = time.perf_counter()
            edge_index = radius_graph(x[:, :2], r=10.0, loop=False)
            if "cuda" in device_name:
                torch.cuda.synchronize()
            end_graph = time.perf_counter()
            
            # 2. Model Computation
            start_model = time.perf_counter()
            _ = runner.model(x, edge_index=edge_index)
            if "cuda" in device_name:
                torch.cuda.synchronize()
            end_model = time.perf_counter()
            
            end_iter = time.perf_counter()
            
            graph_times.append((end_graph - start_graph) * 1000.0)
            model_times.append((end_model - start_model) * 1000.0)
            total_times.append((end_iter - start_iter) * 1000.0)

    avg_graph = np.mean(graph_times)
    std_graph = np.std(graph_times)
    avg_model = np.mean(model_times)
    std_model = np.std(model_times)
    avg_total = np.mean(total_times)
    std_total = np.std(total_times)
    
    print(f"\n--- Timing Breakdown ({device_name}, {current_threads} Threads) ---")
    print(f"{'Component':<20} | {'Mean (ms)':<10} | {'Std (ms)':<10} | {'Percentage':<10}")
    print("-" * 58)
    print(f"{'Graph Generation':<20} | {avg_graph:10.2f} | {std_graph:10.2f} | {(avg_graph/avg_total)*100:9.1f}%")
    print(f"{'Model Computation':<20} | {avg_model:10.2f} | {std_model:10.2f} | {(avg_model/avg_total)*100:9.1f}%")
    print("-" * 58)
    print(f"{'Total (Measured)':<20} | {avg_total:10.2f} | {std_total:10.2f} | 100.0%")
    print(f"Throughput: {1000.0 / avg_total:.2f} Hz")

    # CSV Export
    csv_rows = [
        {
            "Component": "Graph Generation",
            "Mean_ms": f"{avg_graph:.4f}",
            "Std_ms": f"{std_graph:.4f}",
            "Percentage": f"{(avg_graph/avg_total)*100:.2f}%"
        },
        {
            "Component": "Model Computation",
            "Mean_ms": f"{avg_model:.4f}",
            "Std_ms": f"{std_model:.4f}",
            "Percentage": f"{(avg_model/avg_total)*100:.2f}%"
        }
    ]

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

    print(f"--- Setting up SageGNNClassifier RaGNNarok Experiment: {args.config_label} ---")
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../configs/RaGNNarok/{args.config_label}.py"))
    
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
    model = SageGNNClassifier(**model_cfg)

    # Check for checkpoint existence in test-only mode
    if args.test_only and not os.path.exists(args.checkpoint_path):
        print(f"\n[WARNING] Checkpoint not found at {args.checkpoint_path}")
        print("Proceeding with Randomly Initialized weights for demonstration purposes.")
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint_path)
        print(f"Temporary dummy checkpoint saved to {args.checkpoint_path}")

    # Use runner settings identical to training ones
    runner = GNNRunner(
        model=model,
        state_dict_path=args.checkpoint_path,
        cuda_device="cuda:0" if torch.cuda.is_available() else "cpu",
        edge_radius=10.0,
        enable_downsampling=False,
        use_sigmoid=False, # SageGNNClassifier already outputs sigmoid within the model forward pass itself
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
    
    nodes_encoded = input_encoder.encode(nodes)
    
    runner.model.eval()
    with torch.no_grad():
        x = torch.tensor(nodes_encoded, dtype=torch.float32).to(runner.device)
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(x[:, :2], r=10.0, loop=False).to(runner.device)
        
        # We need to manually invoke the model to supply our intermediate arguments
        out, intermediates = runner.model(x, edge_index=edge_index, return_intermediate=True)
        final_predictions = out.cpu().numpy()
        
        # We also need to map the output to layer predictions list to fit into plot_sage_layer_analysis
        layer_predictions = [final_predictions]

    # --- SAGE Layer Analysis Figure ---
    analysis_save_path = os.path.join(args.output_dir, f"{args.config_label}_layer_analysis_idx{args.analysis_idx}.png")
    plotter.plot_sage_layer_analysis(
        nodes=nodes,
        model_outputs=intermediates,
        layer_predictions=layer_predictions,
        gt_labels=labels,
        save_path=analysis_save_path,
        show=False
    )
    print(f"Layer analysis figure saved to {analysis_save_path}")

    # --- PHASE 3: Benchmarking ---
    if not args.skip_benchmark:
        print("\n--- Phase 3: Performance Benchmarking ---")
        
        # 1. GPU Benchmark (Multi-core dispatch)
        if torch.cuda.is_available():
            benchmark_model(runner, dataset, input_encoder, "cuda:0", args.config_label, args.output_dir, num_runs=args.num_runs)
        
        # 2. CPU Multi-core Benchmark (System Default)
        benchmark_model(runner, dataset, input_encoder, "cpu", args.config_label, args.output_dir, num_runs=args.num_runs)

        # 3. CPU Single-core Benchmark
        benchmark_model(runner, dataset, input_encoder, "cpu", args.config_label, args.output_dir, num_threads=1, num_runs=args.num_runs)

    print(f"\n--- Experiment Completed. Results saved to {args.output_dir} ---")

if __name__ == "__main__":
    main()
