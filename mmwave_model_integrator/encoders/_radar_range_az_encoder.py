import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.processors.range_azmith_resp import RangeAzimuthProcessor


class _RadarRangeAzEncoder:
    """Encoder specifically designed to work with raw radar data
    """
    def __init__(self,config_manager:ConfigManager) -> None:
        
        self.config_manager:ConfigManager = config_manager

        #flag to note whether a full encoding is ready or not
        #(for encoders that encode a series of frames)
        self.full_encoding_ready = False
        
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

    def _convert_cartesian_to_spherical(self,points_cart:np.ndarray):
        """Convert an array of points stored as (x,y,z) to (range,azimuth, elevation).
        Note that azimuth = 0 degrees for points on the positive x-axis

        Args:
            points_cart (np.ndarray): Nx3 matrix of points in cartesian (x,y,z)

        Returns:
            (np.ndarray): Nx3 matrix of points in spherical (range, azimuth, elevation) in radians
        """
        ranges = np.sqrt(points_cart[:, 0]**2 + points_cart[:, 1]**2 + points_cart[:, 2]**2)
        azimuths = np.arctan2(points_cart[:, 1], points_cart[:, 0])
        elevations = np.arccos(points_cart[:, 2] / ranges)

        return  np.column_stack((ranges,azimuths,elevations))
        
    def _convert_spherical_to_cartesian(self,points_spherical:np.ndarray):
        """Convert an array of points stored as (range, azimuth, elevation) to (x,y,z)

        Args:
            points_spherical (np.ndarray): Nx3 matrix of points in spherical (range,azimuth, elevation)

        Returns:
            (np.ndarray): Nx3 matrix of  points in cartesian (x,y,z)
        """

        x = points_spherical[:,0] * np.sin(points_spherical[:,2]) * np.cos(points_spherical[:,1])
        y = points_spherical[:,0] * np.sin(points_spherical[:,2]) * np.sin(points_spherical[:,1])
        z = points_spherical[:,0] * np.cos(points_spherical[:,2])


        return np.column_stack((x,y,z))