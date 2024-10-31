import numpy as np

from mmwave_model_integrator.output_encoders._lidar_2D_pc_encoder import _Lidar2DPCEncoder
from mmwave_model_integrator.transforms.coordinate_transforms import cartesian_to_spherical,polar_to_cartesian

class RadCloudOutputEncoder(_Lidar2DPCEncoder):

    def __init__(
            self,
            max_range_m:float = 8.56,
            num_range_bins:int = 64,
            angle_range_rad:list=[-np.pi/2 - 0.87,-np.pi/2 + 0.87],
            num_angle_bins:int = 48,
            num_previous_frames=0
    ):
        
        self.max_range_m:float = max_range_m
        self.num_range_bins:int = num_range_bins
        
        self.angle_range_rad:list = angle_range_rad
        self.num_angle_bins:int = num_angle_bins

        self.num_previous_frames:int = num_previous_frames

        super().__init__()
    
    def configure(self):

        #initialize the range bins
        self.angle_bins_rad = np.linspace(
                start=self.angle_range_rad[0],
                stop=self.angle_range_rad[1],
                num=self.num_angle_bins
            )

        #initialize the angle bins
        self.range_bins_m = np.linspace(
            start=0,
            stop=self.max_range_m,
            num=self.num_range_bins
            )

        return



    def encode(self, lidar_pc: np.ndarray) -> np.ndarray:
        """Encodes data for the radcloud model

        Args:
            lidar_pc (np.ndarray): N x 3 3D point cloud of lidar data

        Returns:
            np.ndarray: Quantized grid as the ground truth output of the model
        """

        #remove the ground plane
        pc = self._remove_ground_plane(lidar_pc)

        #convert points to spherical
        pc = cartesian_to_spherical(pc)

        #filter ranges and azimuths of detections
        pc = self._filter_ranges_and_azimuths(pc)

        #convert the points to a quantized grid
        grid = self.points_polar_to_grid(pc)

        #perform BCC to remove miscellaneous smaller detections
        grid = self._apply_binary_connected_component_analysis_to_grid(grid)

        return grid