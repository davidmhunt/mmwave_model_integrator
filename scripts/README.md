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

## Macro-Reasoning Experiment (Hierarchical GNN)

The `run_macro_reasoning_experiment.py` script is designed for the `MacroReasoningGnn` architecture, which includes a secondary GNN layer for super-nodes to understand their spatial distribution before attention.

### Running the Macro-Reasoning Experiment

```bash
python run_macro_reasoning_experiment.py --test_only
```

### Unique Features:
- **Smart Super-Nodes Plot**: The forward pass analysis now includes a 6th panel (in a 2x3 grid) specifically visualizing the activations of the macro-reasoning GNN.
- **Persistent Naming**: Resulting `.png` files are automatically prefixed with the `config_label` (e.g., `IcaRAus_MacroReasoningGnn_analysis_idx1000.png`) to prevent overwriting results from other architectures.
