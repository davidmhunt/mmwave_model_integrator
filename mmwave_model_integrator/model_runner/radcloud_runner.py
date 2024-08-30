import numpy as np
import torch
from torch.nn import Module
from torchvision import transforms
from torchvision.transforms import Compose

from radcloud.models.unet import unet

from mmwave_model_integrator.model_runner._model_runner import _ModelRunner

class RadCloudRunner(_ModelRunner):

    def __init__(
            self,
            state_dict_path: str,
            cuda_device="cuda:0") -> None:
        
        #specify the radcloud model
        radcloud_model = unet(
            encoder_input_channels= 40,
            encoder_out_channels= (64,128,256),
            decoder_input_channels= (512,256,128),
            decoder_out_channels= 64,
            output_channels= 1,
            retain_dimmension= False,
            input_dimmensions= (64,48)
        )

        #specify the transforms
        self.transforms = Compose([
            transforms.ToTensor(),
            transforms.Resize((64,48))
        ])
        
        super().__init__(
            model=radcloud_model,
            state_dict_path=state_dict_path,
            cuda_device=cuda_device
        )

    def configure(self):
        """Load the model onto the desired device. 
        Same as parent class, no additional function needed
        """
        #no additional function needs to be added
        super().configure()

        return

    def make_prediction(self, input: np.ndarray):

        x = input.astype(np.float32)

        with torch.no_grad():
        
            #apply transforms (i.e: convert to tensor)
            self.model.eval()
            
            x = self.transforms(x)
            
            #since only one sample, need to unsqueeze
            x = torch.unsqueeze(x,0)

            #send x to device
            x = x.to(self.device)

            #get the prediction and apply sigmoid
            pred = self.model(x).squeeze()
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()

            #filter out weak predictions
            pred = (pred > 0.5) * 1.0

        return pred