
import optuna
import os
import sys
import torch
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from mmwave_model_integrator.config import Config
import mmwave_model_integrator.torch_training.trainers as trainers

# Set up paths
sys.path.append("../")

def objective(trial):
    # Load base config
    config_path = "../../configs/IcaRAus_gnn_base.py"
    config = Config(config_path)

    # Suggest hyperparameters
    k_val = trial.suggest_int("k", 10, 40, step=5)
    hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Override config with trial parameters
    config.model['k'] = k_val
    config.model['hidden_channels'] = hidden_channels
    config.trainer['optimizer']['lr'] = lr
    config.trainer['optimizer']['weight_decay'] = weight_decay
    
    # Also update model K inside the trainer config if it's passed separately or used there
    config.trainer['model']['k'] = k_val
    config.trainer['model']['hidden_channels'] = hidden_channels

    # Unique working directory for this trial
    trial_dir = os.path.join("tuning_logs", f"trial_{trial.number}")
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
    config.trainer['working_dir'] = trial_dir
    config.trainer['save_name'] = f"model_trial_{trial.number}"

    try:
        # Initialize Trainer
        trainer_config = config.trainer
        trainer_class = getattr(trainers, trainer_config.pop('type'))
        trainer = trainer_class(**trainer_config)

        # Run training
        best_val_loss = trainer.train_model(trial=trial)

        return best_val_loss
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise e
    finally:
        # cleanup to prevent OOM
        if 'trainer' in locals():
            del trainer
        if 'config' in locals():
            del config
        import gc
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # Create study
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    # Run optimization
    study.optimize(objective, n_trials=2)

    # Print best results
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Generate visualizations
    print("Generating visualizations...")
    os.makedirs("tuning_logs", exist_ok=True)
    
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_html("tuning_logs/optimization_history.html")
        
        fig2 = plot_param_importances(study)
        fig2.write_html("tuning_logs/param_importances.html")
        
        fig3 = plot_parallel_coordinate(study)
        fig3.write_html("tuning_logs/parallel_coordinate.html")
        print("Visualizations saved to tuning_logs/")
    except Exception as e:
        print(f"Visualization failed: {e}")
