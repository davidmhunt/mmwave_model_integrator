import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.processors.range_azmith_resp import RangeAzimuthProcessor

from mmwave_model_integrator.encoders._radar_range_az_encoder import _RadarRangeAzEncoder

class RadCloudEncoder(_RadarRangeAzEncoder):

    def __init__(
            self,
            config_manager: ConfigManager,
            max_range_bin:int,
            num_chirps_to_encode:int,
            radar_fov_rad:list,
            range_az_num_angle_bins:int,
            power_range_dB:list) -> None:
        
        #configuration parameters
        self.max_range_bin:int = max_range_bin
        self.num_chirps_to_encode:int = num_chirps_to_encode
        self.radar_fov_rad:list = radar_fov_rad
        self.range_az_num_angle_bins:int = range_az_num_angle_bins
        self.power_range_dB:list = power_range_dB

        #derrived parameters
        self.angle_bins_to_keep:np.ndarray = None

        super().__init__(config_manager)

        return
    
    def configure(self):

        #configure virtual array processors and range az response
        super().configure()

        #determine the finalized set of range bins
        self.range_bins = \
            self.range_azimuth_processor.range_bins[:self.max_range_bin]

        #determine the angle bins to keep
        self.angle_bins_to_keep = \
            (self.range_azimuth_processor.angle_bins > self.radar_fov_rad[0]) \
            & (self.range_azimuth_processor.angle_bins < self.radar_fov_rad[1])
        self.angle_bins = \
            self.range_azimuth_processor.angle_bins[self.angle_bins_to_keep]
        #compute the mesh grid
        self.thetas,self.rhos = \
            np.meshgrid(self.range_azimuth_processor.angle_bins[self.angle_bins_to_keep],
                        self.range_azimuth_processor.range_bins[:self.max_range_bin])
        self.x_s = np.multiply(self.rhos,np.sin(self.thetas))
        self.y_s = np.multiply(self.rhos,np.cos(self.thetas))
        
        return
    
    def encode(self, adc_data_cube: np.ndarray) -> np.ndarray:

        frame_range_az_heatmap = np.zeros(
            shape=(
                self.max_range_bin,
                np.sum(self.angle_bins_to_keep),
                self.num_chirps_to_encode
            )
        )

        #process the adc cube if virtual arrays were used
        adc_data_cube = self.virtual_array_reformater.process(adc_data_cube)

        for i in range(self.num_chirps_to_encode):

            #compute the full range azimuth response
            #(returns magnitude of the response)
            rng_az_resp = self.range_azimuth_processor.process(
                adc_cube=adc_data_cube,
                chirp_idx=i)
            
            #convert to dB
            rng_az_resp = 20 * np.log10(rng_az_resp)

            #filter to only desired ranges and angles
            rng_az_resp = rng_az_resp[
                :self.max_range_bin,
                self.angle_bins_to_keep]

            #threshold the input data
            rng_az_resp[rng_az_resp <= self.power_range_dB[0]] = \
                self.power_range_dB[0]
            rng_az_resp[rng_az_resp >= self.power_range_dB[1]] = \
                self.power_range_dB[1]
            
            #normalize the input data to be between 0 and 1
            rng_az_resp = (rng_az_resp - self.power_range_dB[0]) / \
                (self.power_range_dB[1] - self.power_range_dB[0])

            frame_range_az_heatmap[:,:,i] = rng_az_resp

        return frame_range_az_heatmap