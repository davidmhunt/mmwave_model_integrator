
import optuna
import os
import sys
import torch
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from mmwave_model_integrator.config import Config
import mmwave_model_integrator.torch_training.trainers as trainers

# Set up paths
sys.path.append("../")

config_name = "IcaRAus_gnn_two_stream"

def objective(trial):
    # Load base config
    config_path = "../../configs/IcaRAus_gnn/{}.py".format(config_name)
    config = Config(config_path)

    # Suggest hyperparameters
    k_val = trial.suggest_int("k", 10, 40, step=5)
    hidden_channels = trial.suggest_int("hidden_channels",16,32,step=4)
    # hidden_channels = trial.suggest_categorical("hidden_channels", [16, 24, 32])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.7)

    # Override config with trial parameters
    config.model['k'] = k_val
    config.model['hidden_channels'] = hidden_channels
    config.model['dropout'] = dropout
    config.trainer['optimizer']['lr'] = lr
    config.trainer['optimizer']['weight_decay'] = weight_decay
    
    # Also update model params inside the trainer config if it's passed separately or used there
    config.trainer['model']['k'] = k_val
    config.trainer['model']['hidden_channels'] = hidden_channels
    config.trainer['model']['dropout'] = dropout

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
    
    # Ensure the directory exists for the database
    log_dir = "tuning_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Database URL for optuna-dashboard support
    db_url = f"sqlite:///{log_dir}/optuna.db"
    
    # Create study
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize", 
        pruner=pruner,
        storage=db_url,
        study_name="{}_optimization".format(config_name),
        load_if_exists=True
    )
    
    print(f"Study created. Run dashboard with: optuna-dashboard {db_url}")
    
    # Run optimization
    study.optimize(objective, n_trials=50)

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
