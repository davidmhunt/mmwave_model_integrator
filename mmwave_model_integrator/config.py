import os
import types

class Config:
    def __init__(self, config_path):
        self.load_config(config_path)

    def load_config(self, config_path):
        # Read the content of the config file
        with open(config_path, 'r') as file:
            exec(file.read(), globals())
        
        # If there is a _base_ key, load the base configurations
        if '_base_' in globals():
            base_files = globals()['_base_']
            for base_file in base_files:
                self.load_config(os.path.join(os.path.dirname(config_path), base_file))
        
        # Initialize class attributes using setattr
        for key, value in globals().items():
            if not key.startswith('__') and not callable(value) and key != '_base':
                setattr(self, key, value)

    def print_config(self, config=None, indent=0):
        # Print the configuration in a nice format with indentation for nested dictionaries
        if config is None:
            config = self.__dict__

        for key, value in config.items():
            if not isinstance(value,types.ModuleType):
                print(' ' * indent + f"{key}=", end="")
                if isinstance(value, dict):
                    print("dict(")
                    self.print_config(value, indent + 4)
                    print(' ' * indent + "),")
                else:
                    print("{},".format(value))