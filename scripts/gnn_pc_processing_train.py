from mmwave_model_integrator.config import Config
import mmwave_model_integrator.torch_training.trainers as trainers

config_path = "../configs/sage_gnn_cylindrical.py"

config_paths = [
    "../configs/sage_gnn_cylindrical_10fh.py",
    "../configs/sage_gnn_cylindrical_20fh.py",
    "../configs/sage_gnn_cylindrical_30fh.py",
    "../configs/sage_gnn_cylindrical_40fh.py"
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