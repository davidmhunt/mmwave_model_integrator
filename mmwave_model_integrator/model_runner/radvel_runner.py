import numpy as np
import torch
from torch.nn import Module
from torchvision.transforms import Compose
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner

class RadVelRunner(_ModelRunner):

    def __init__(
            self, 
            model, 
            state_dict_path, 
            cuda_device="cuda:0",
            transforms:list=[]):
        
        self.transforms = []

        super().__init__(model, state_dict_path, cuda_device)
    
    def configure(self):
        """Load the model onto the desired device. 
        Same as parent class, no additional function needed
        """
        #no additional function needs to be added
        super().configure()

        return

    def make_prediction(self, input: np.ndarray):

        # Convert to PyTorch tensors
        input = input.astype(np.float32)
        if input.ndim == 2:  # [H, W]
            input = torch.from_numpy(input).unsqueeze(0)  # -> [1, H, W]
        elif input.ndim == 3:
            input = torch.from_numpy(input)  # Already [C, H, W]
        else:
            raise ValueError(f"Unexpected image shape: {input.shape}")
        
        if len(self.transforms) > 0:
            transform = Compose(self.transforms)
            input = transform(input)
        
        x = torch.unsqueeze(input,0)
        x = x.to(self.device)

        #perform the forard pass
        pred = self.model(x).squeeze()
        pred = pred.cpu().detach().numpy()
        
        return pred