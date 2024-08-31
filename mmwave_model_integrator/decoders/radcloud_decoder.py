import numpy as np

from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder

class RadCloudDecoder(_lidarPCPolarDecoder):

    def __init__(self,
                 max_range_m:float,
                 num_range_bins:int,
                 angle_range_rad:list,
                 num_angle_bins:int) -> None:

        #additional range parameters
        self.max_range_m = max_range_m
        self.num_range_bins = num_range_bins
        self.range_res_m = 0.0

        #additional angle parameters
        self.az_angle_range_rad:list = angle_range_rad
        self.num_angle_bins = num_angle_bins
        self.az_angle_res_rad = 0.0
        self.az_angle_res_deg = 0.0

        super().__init__()
    
    def configure(self):

        #configure angle parameters
        self.az_angle_res_rad = \
            (self.az_angle_range_rad[1] - self.az_angle_range_rad[0]) / \
                self.num_angle_bins
        self.az_angle_res_deg = np.rad2deg(self.az_angle_res_rad)

        self.angle_bins = \
            np.flip(
                np.arange(
                    self.az_angle_range_rad[0],
                    self.az_angle_range_rad[1],
                    self.az_angle_res_rad))
        
        #configure range bins
        self.range_res_m = self.max_range_m/self.num_range_bins

        self.range_bins = np.flip(np.arange(0,self.max_range_m,self.range_res_m))

        return super().configure()
    
    def decode(self, model_prediction: np.ndarray) -> np.ndarray:

        #get the nonzero coordinates
        rng_idx,az_idx = np.nonzero(model_prediction)

        rng_vals = self.range_bins[rng_idx]
        az_vals = self.angle_bins[az_idx]

        return np.column_stack((rng_vals,az_vals))