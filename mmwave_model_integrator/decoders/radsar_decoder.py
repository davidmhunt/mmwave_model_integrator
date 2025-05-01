import numpy as np

from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder

class RadSARDecoder(_lidarPCPolarDecoder):

    def __init__(self,
                 max_range_m:float,
                 num_range_bins:int,
                 angle_range_rad:list,
                 num_angle_bins:int) -> None:

        #additional range parameters
        self.max_range_m = max_range_m
        self.num_range_bins = num_range_bins

        #additional angle parameters
        self.az_angle_range_rad:list = angle_range_rad
        self.num_angle_bins = num_angle_bins

        super().__init__()
    
    def configure(self):

        self.angle_bins = np.linspace(
            start=self.az_angle_range_rad[0],
            stop=self.az_angle_range_rad[1],
            num=self.num_angle_bins
        )

        self.range_bins = np.linspace(
            start=0,
            stop=self.max_range_m,
            num=self.num_range_bins
        )

        return super().configure()
    
    def decode(self, model_prediction: np.ndarray) -> np.ndarray:

        #get the nonzero coordinates
        rng_idx,az_idx = np.nonzero(model_prediction)

        rng_vals = self.range_bins[rng_idx]
        az_vals = self.angle_bins[az_idx]

        return np.column_stack((rng_vals,az_vals))