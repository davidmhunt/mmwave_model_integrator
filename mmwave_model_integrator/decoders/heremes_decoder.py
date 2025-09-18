import numpy as np

from mmwave_model_integrator.decoders._decoder import _Decoder
from mmwave_model_integrator.input_encoders.hermes_encoder import HermesEncoder

class HermesDecoder(_Decoder):

    def __init__(self,
                 input_encoder: HermesEncoder,
                 output_size:np.array=(96,88)
                 ) -> None:

        #additional range parameters
        self.input_encoder = input_encoder
        self.output_size = output_size
        
        self.x_vals_m = None
        self.y_vals_m = None

        self.x_mesh = None
        self.y_mesh = None

        super().__init__()
    
    def configure(self):

        x_min = np.min(self.input_encoder.interp_x_s)
        x_max = np.max(self.input_encoder.interp_x_s)
        y_min = np.min(self.input_encoder.interp_y_s)
        y_max = np.max(self.input_encoder.interp_y_s)

        self.x_vals_m = np.linspace(
            start=x_min,
            stop=x_max,
            num=self.output_size[0]
        )

        self.y_vals_m = np.linspace(
            start=y_min,
            stop=y_max,
            num=self.output_size[1]
        )

        # Create a meshgrid from the linspaces
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_vals_m, self.y_vals_m, indexing='ij')

        return super().configure()
    
    def decode(self, model_prediction: np.ndarray) -> np.ndarray:

        #get the nonzero coordinates
        x_idx,y_idx = np.nonzero(model_prediction)

        x_vals = self.x_vals_m[x_idx]
        y_vals = self.y_vals_m[y_idx]

        return np.column_stack((x_vals,y_vals))