import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.processors.doppler_azimuth_resp import DopplerAzimuthProcessor
from mmwave_model_integrator.input_encoders._input_encoder import _InputEncoder

from mmwave_model_integrator.transforms import coordinate_transforms

class _DopplerAzEncoder(_InputEncoder):

    def __init__(self,
                 config_manager:ConfigManager,
                 num_angle_bins:int=64) -> None:

        self.config_manager:ConfigManager = config_manager
        self.num_angle_bins = num_angle_bins

        #response processing
        self.virtual_array_reformater:VirtualArrayReformatter = None
        self.doppler_azimuth_processor:DopplerAzimuthProcessor = None

        #assistance for plotting
        self.angle_bins:np.ndarray = None
        self.vel_bins:np.ndarray = None

        #array for the encoded data
        #NOTE: #indexed/implemented depending on child class
        self.encoded_data:np.ndarray = None
        
        #complete the configuration
        super().__init__()

        return
    
    def configure(self):
        """Configure the doppler azimuth processor and virtual array
        processor. Remaining functionality must be implemented by 
        child class to configure its modules.
        """

        #configure range azimuth processor
        self.doppler_azimuth_processor = DopplerAzimuthProcessor(
            config_manager=self.config_manager,
            num_angle_bins=self.num_angle_bins
        )

        #configure virtual array reformatter
        self.virtual_array_reformater = VirtualArrayReformatter(
            config_manager=self.config_manager)
        
        #configure range and angle bins
        self.vel_bins = self.doppler_azimuth_processor.vel_bins
        self.angle_bins = self.doppler_azimuth_processor.angle_bins
        
        return
    
    def encode(self,adc_data_cube:np.ndarray)->np.ndarray:
        """Implemented by child class to encode data for a specific
        model

        Args:
            adc_data_cube (np.ndarray): (rx antennas) x (adc samples) x
                (num_chirps) adc data cube consisting of complex data

        Returns:
            np.ndarray: np.ndarray consisting of data to be input
                into the model
        """
        pass

    def reset(self):
        """Implemented by child class to reset encoder history (if applicable)
        """
        self.full_encoding_ready = False
        return