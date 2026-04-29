import argparse
import os
import sys
import yaml
import importlib

# Centralized Experiment Registry
# Maps standardized experiment names to their respective modules and default configurations
EXPERIMENT_REGISTRY = {
    "IcaRAus_gnn_IcaRAus_ds": {
        "module_name": "run_IcaRAus_experiment_IcaRAus_ds",
        "default_mapping": "IcaRAus_gnn_IcaRAus_ds_folds.yaml",
        "default_base_config": "IcaRAus_gnn_final_IcaRAus_ds"
    },
    "IcaRAus_gnn_RaGNNarok_ds": {
        "module_name": "run_IcaRAus_experiment_RaGNNarok_ds_and_format",
        "default_mapping": "IcaRAus_gnn_RaGNNarok_ds_folds.yaml",
        "default_base_config": "IcaRAus_gnn_final_RaGNNarok_ds_and_format"
    },
    "RaGNNarok_gnn_IcaRAus_ds": {
        "module_name": "run_RaGNNarok_experiment_IcaRAus_ds_and_format",
        "default_mapping": "RaGNNarok_gnn_IcaRAus_ds_folds.yaml",
        "default_base_config": "RaGNNarok_final_IcaRAus_ds_and_format"
    },
    "RaGNNarok_gnn_RaGNNarok_ds": {
        "module_name": "run_RaGNNarok_experiment_RaGNNarok_ds",
        "default_mapping": "RaGNNarok_gnn_RaGNNarok_ds_folds.yaml",
        "default_base_config": "RaGNNarok_final_RaGNNarok_ds"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Multi-Dataset GNN Experiment Runner")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENT_REGISTRY.keys()),
                        help="List of experiments to run. If omitted, all supported experiments will be listed.")
    
    # Standard overrides for all selected experiments
    parser.add_argument("--test_only", action="store_true", help="Skip training and only run evaluation.")
    parser.add_argument("--num_instances", type=int, default=5, help="Number of instances to plot.")
    parser.add_argument("--analysis_idx", type=int, default=1000, help="Index for layer-wise analysis.")
    parser.add_argument("--print_stats", action="store_true", help="Print stats during inference.")
    parser.add_argument("--skip_benchmark", action="store_true", help="Skip benchmarking phase.")
    
    # Overrides for specific mapping or base config (apply to all selected experiments)
    parser.add_argument("--mapping_file_override", type=str, help="Override default mapping file.")
    parser.add_argument("--base_config_override", type=str, help="Override default base configuration template.")
    
    return parser.parse_args()

def resolve_mapping_path(mapping_name):
    """Resolves mapping file path locally or in experiment_configs/."""
    if os.path.exists(mapping_name):
        return mapping_name
    
    # Try experiment_configs/
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(scripts_dir, "experiment_configs", mapping_name)
    if not mapping_name.endswith(".yaml") and not os.path.exists(alt_path):
        alt_path += ".yaml"
    
    if os.path.exists(alt_path):
        return alt_path
    
    return None

def main():
    args = parse_args()

    if not args.experiments:
        print("\nAvailable Experiments:")
        for name in EXPERIMENT_REGISTRY:
            print(f"  - {name}")
        print("\nPlease specify one or more experiments using --experiments <name1> <name2> ...")
        return

    for exp_name in args.experiments:
        print(f"\n" + "="*80)
        print(f" STARTING EXPERIMENT SUITE: {exp_name}")
        print("="*80)

        exp_info = EXPERIMENT_REGISTRY[exp_name]
        
        # 1. Dynamically import the experiment module
        try:
            module = importlib.import_module(exp_info["module_name"])
            run_cycle = getattr(module, "run_experiment_cycle")
        except (ImportError, AttributeError) as e:
            print(f"Error: Could not load module or run_experiment_cycle for {exp_name}: {e}")
            continue

        # 2. Resolve mapping file
        mapping_file = args.mapping_file_override if args.mapping_file_override else exp_info["default_mapping"]
        mapping_path = resolve_mapping_path(mapping_file)
        
        if not mapping_path:
            print(f"Error: Mapping file {mapping_file} not found.")
            continue

        with open(mapping_path, "r") as f:
            mapping = yaml.safe_load(f)

        if not isinstance(mapping, dict):
            print(f"Error: Mapping file {mapping_path} must be a YAML dictionary.")
            continue

        # 3. Resolve base config
        base_config = args.base_config_override if args.base_config_override else exp_info["default_base_config"]

        # 4. Iterate over folds in the mapping
        for config_label, dataset_path in mapping.items():
            print(f"\nProcessing Fold: {config_label}")
            print(f"Dataset Path: {dataset_path}")
            print(f"Using Base Config: {base_config}")
            print("-" * 40)

            # Resolve paths for this fold
            scripts_dir = os.path.dirname(os.path.abspath(__file__))
            if "IcaRAus_gnn" in exp_name:
                checkpoint_dir = os.path.join(scripts_dir, "working_dir", "IcaRAus_gnn")
            else:
                checkpoint_dir = os.path.join(scripts_dir, "working_dir", "RaGNNarok")

            checkpoint_path = os.path.join(checkpoint_dir, f"{config_label}.pth")
            output_dir = os.path.join("experiment_results", config_label)

            success = run_cycle(
                config_label=config_label,
                dataset_path=dataset_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                base_config_label=base_config,
                test_only=args.test_only,
                num_instances=args.num_instances,
                analysis_idx=args.analysis_idx,
                print_stats=args.print_stats,
                skip_benchmark=args.skip_benchmark
            )
            
            if not success:
                print(f"Warning: Experiment cycle failed for fold {config_label}")

    print("\n" + "="*80)
    print(" ALL REQUESTED EXPERIMENT SUITES COMPLETED.")
    print("="*80)

if __name__ == "__main__":
    main()
