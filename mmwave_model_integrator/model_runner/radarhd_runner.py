import numpy as np
import torch
from torch.nn import Module
from torchvision import transforms
from torchvision.transforms import Compose

from radarhd.model import UNet1
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner

class RadarHDRunner(_ModelRunner):

    def __init__(
            self,
            state_dict_path: str,
            cuda_device="cuda:0") -> None:
        
        #specify the radcloud model
        radarhd_model = UNet1(
            n_channels=41,
            n_classes=1,
            bilinear=True
        )

        #specify the transforms
        self.transforms = Compose([
            transforms.ToTensor()
        ])
        
        super().__init__(
            model=radarhd_model,
            state_dict_path=state_dict_path,
            cuda_device=cuda_device
        )

    def configure(self):
        """Load the model onto the desired device. 
        overriding parent class
        """
        #no additional function needs to be added
        #load the state dictionary
        if self.device != 'cpu':
            self.model.load_state_dict(
                torch.load(self.state_dict_path,
                           weights_only=True)["state_dict"])
        else:
            self.model.load_state_dict(
                torch.load(
                    self.state_dict_path,
                    map_location='cpu',
                    weights_only=True)["state_dict"])
        
        #send the model to the device
        self.model.to(self.device)

        #put the model into evaluation mode
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")

        return

    def make_prediction(self, input: np.ndarray):

        x = input.astype(np.float32)
        x = x / 255.0

        with torch.no_grad():
        
            #apply transforms (i.e: convert to tensor)
            self.model.eval()
            
            # x = self.transforms(x)
            x = torch.Tensor(x)
            
            #since only one sample, need to unsqueeze
            x = torch.unsqueeze(x,0)

            #send x to device
            x = x.to(self.device)

            #get the prediction and apply sigmoid
            pred = self.model(x).squeeze()
            pred = pred.cpu().numpy()

            #filter out weak predictions
            pred = (pred*255).astype(np.uint8)

            #return all predictions with a value greater than 1
            pred = (pred >= 1) * 1

        return pred