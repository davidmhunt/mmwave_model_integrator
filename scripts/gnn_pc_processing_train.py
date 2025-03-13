from mmwave_model_integrator.config import Config
import mmwave_model_integrator.torch_training.trainers as trainers

config_paths = [
    "../configs/RaGNNarok/RaGNNarok_20_fh_edge_rad_5.py",
]

def train_model(config_path):
    config = Config(config_path)

    config.print_config()

    trainer_config = config.trainer
    trainer_class = getattr(trainers,trainer_config.pop('type'))
    trainer = trainer_class(**trainer_config)
    trainer.train_model()

    return

if __name__ == "__main__":

    for config_path in config_paths:
        train_model(config_path)