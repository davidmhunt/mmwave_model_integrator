import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.processors.doppler_azimuth_resp import DopplerAzimuthProcessor

from mmwave_model_integrator.input_encoders._doppler_az_encoder import _DopplerAzEncoder

class RadVelEncoder(_DopplerAzEncoder):

    def __init__(
            self,
            config_manager: ConfigManager,
            num_angle_bins:int,
            min_power_threshold_dB:float = 40
            ) -> None:
        
        #configuration parameters
        self.min_power_threshold_dB:float = min_power_threshold_dB

        #derrived parameters
        self.angle_bins_to_keep:np.ndarray = None

        #array for latest encoded data (from parent class)
        #NOTE: indexed by range bin, az bin, chirp idx
        self.encoded_data:np.ndarray = None

        super().__init__(
            config_manager=config_manager,
            num_angle_bins=num_angle_bins
        )

        return
    
    def configure(self):

        #configure virtual array processors and range az response
        return super().configure()
            
    def encode(self, adc_data_cube: np.ndarray) -> np.ndarray:

        #process the virtual arrays
        adc_data_cube = self.virtual_array_reformater.process(adc_data_cube)

        #compute the raw response (already an magnitude of the average across all samples)
        #indexed by velocity,antenna
        resp = self.doppler_azimuth_processor.process(adc_cube=adc_data_cube)
        
        #convert to dB
        resp = 20 * np.log10(resp)
        #remove anything below the min_threshold dB down
        thresholded_val = np.max(resp) - self.min_power_threshold_dB
        idxs = resp <= thresholded_val
        resp[idxs] = thresholded_val

        #normalize the response data to be between 0 and 1
        self.encoded_data = (resp - thresholded_val)/self.min_power_threshold_dB

        self.full_encoding_ready = True

        return self.encoded_data
    
    def reset(self):
        """No history, method isn't used for this encoder
        """
        return super().reset()