import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.processors.range_angle_resp import RangeAngleProcessor
from mmwave_model_integrator.input_encoders._input_encoder import _InputEncoder

from mmwave_model_integrator.transforms import coordinate_transforms

class _RadarRangeAzEncoder(_InputEncoder):
    """Encoder specifically designed to work with raw radar data
    """
    def __init__(self,config_manager:ConfigManager) -> None:
        
        self.config_manager:ConfigManager = config_manager
        
        #mesh grids for polar and cartesian plotting - SET BY CHILD CLASS
        self.thetas:np.ndarray = None
        self.rhos:np.ndarray = None
        self.x_s:np.ndarray = None
        self.y_s:np.ndarray = None

        #range and angle bins - SET BY CHILD CLASS
        if not self.num_az_angle_bins:
            self.num_az_angle_bins = None
        self.range_bins:np.ndarray = None
        self.angle_bins:np.ndarray = None

        #response processing
        self.virtual_array_reformater:VirtualArrayReformatter = None
        self.range_azimuth_processor:RangeAngleProcessor = None

        #array for the encoded data
        #NOTE: #indexed/implemented depending on child class
        self.encoded_data:np.ndarray = None
        
        #complete the configuration
        super().__init__()

        return

    def configure(self):
        """Configure the range azimuth processor and virtual array
        processor. Remaining functionality must be implemented by 
        child class to configure its modules.
        """

        #configure range azimuth processor
        self.range_azimuth_processor = RangeAngleProcessor(
            config_manager=self.config_manager,
            num_angle_bins=self.num_az_angle_bins
        )

        #configure virtual array reformatter
        self.virtual_array_reformater = VirtualArrayReformatter(
            config_manager=self.config_manager)
        
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

    def get_rng_az_resp_from_encoding(self,rng_az_resp:np.ndarray)->np.ndarray:
        """Given an encoded range azimuth response, return a single
        range azimuth response that can then be plotted. Implemented
        by child class

        Args:
            rng_az_resp (np.ndarray): encoded range azimuth response

        Returns:
            np.ndarray: (range bins) x (az bins) range azimuth response
        """

        pass