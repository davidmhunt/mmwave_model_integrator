import numpy as np
import torch
from torch.nn import Module

class _ModelRunner:

    def __init__(
            self,
            model:Module,
            state_dict_path:str,
            cuda_device = "cuda:0") -> None:
        
        if torch.cuda.is_available() and cuda_device!="cpu":

            self.device = cuda_device
            torch.cuda.set_device(self.device)
            print("_ModelRunner: using GPU: {}".format(cuda_device))
        else:
            self.device = "cpu"
            print("_ModelRunner: using CPU")
        
        self.state_dict_path = state_dict_path
        self.model:Module = model

        self.configure()
    
    def configure(self):
        """Load the model onto the desired device. Remaining
        behavior implemented by child classes
        """
        #load the state dictionary
        if self.device != 'cpu':
            self.model.load_state_dict(
                torch.load(self.state_dict_path,
                           weights_only=True))
        else:
            self.model.load_state_dict(
                torch.load(
                    self.state_dict_path,
                    map_location='cpu',
                    weights_only=True))
        
        #send the model to the device
        self.model.to(self.device)

        #put the model into evaluation mode
        self.model.eval()

        #TODO: Remaining behavior implemented by child classes
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        return

    def make_prediction(self, input:np.ndarray):
        """Function to make a prediction using the given model.
        Implemented by the child classes

        Args:
            input (np.ndarray): model input expressed as a numpy array
        """

        pass