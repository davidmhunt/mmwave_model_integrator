import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.processors.range_azmith_resp import RangeAzimuthProcessor


class _RadarRangeAzEncoder:
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
        self.range_bins:np.ndarray = None
        self.angle_bins:np.ndarray = None

        #response processing
        self.virtual_array_reformater:VirtualArrayReformatter = None
        self.range_azimuth_processor:RangeAzimuthProcessor = None

        #complete the configuration
        self.configure()

        return

    def configure(self):
        """Configure the range azimuth processor and virtual array
        processor. Remaining functionality must be implemented by 
        child class to configure its modules.
        """

        #configure range azimuth processor
        self.range_azimuth_processor = RangeAzimuthProcessor(
            config_manager=self.config_manager,
            num_angle_bins=self.range_az_num_angle_bins
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