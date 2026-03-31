# IcaRAus GNN Cross-Attention Experiment

This directory contains the `run_IcaRAus_gnn_cross_attention_experiment.py` script, which provides a unified interface for training, evaluating, and visualizing the Cross-Attention GNN model.

## Features

1. **Unified Pipeline**: Train and evaluate the model in a single execution.
2. **Multi-Instance Plotting**: Automatically generates a configurable grid (e.g., 5x3) comparing the Input, Ground Truth, and Predictions for a random subset of validation/test samples.
3. **Forward Pass Analysis**: Performs an ablation study by visualizing intermediate outputs from the cross-attention model during a forward pass. This helps analyze:
    - Point Cloud Local Feature Activations
    - Selected Super-nodes (via FPS)
    - Cumulative Cross-Attention Focus Maps

## Usage

Ensure your environment is active (e.g., `eval $(poetry env activate)` in the `odometry` root).

### 1. Full Pipeline (Train -> Evaluate -> Visualize)
To trigger the full pipeline using the default configuration (`IcaRAus_gnn_cross_attention_gnn.py`):

```bash
python run_IcaRAus_gnn_cross_attention_experiment.py
```

### 2. Evaluation Only (Skip Training)
If you already have a trained checkpoint and want to regenerate the plots or analyze a different sample:

```bash
python run_IcaRAus_gnn_cross_attention_experiment.py --test_only
```

### Customizing Paths and Parameters
The script accepts several command-line arguments to override defaults:

- `--config_label`: The base name of your config file in `../configs/IcaRAus_gnn/` (e.g., `IcaRAus_gnn_cross_attention_gnn`).
- `--dataset_path`: Override the path to your generated dataset.
- `--checkpoint_path`: The file path to load/save the PyTorch model checkpoint (`.pth`).
- `--num_instances`: The number of rows for the multi-instance comparison plot. Default is `5` (yielding a 5x3 figure).
- `--output_dir`: The directory where generated `.png` plots will be saved. Default is `./experiment_results`.
- `--analysis_idx`: The dataset index to pick for the detailed forward-pass analysis. Default is `1000`.

**Example:**
```bash
python run_IcaRAus_gnn_cross_attention_experiment.py --test_only --num_instances 3 --analysis_idx 500 --output_dir ./my_awesome_results
```
