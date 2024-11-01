import numpy as np

from mmwave_model_integrator.ground_truth_encoders._gt_encoder_lidar2D import _GTEncoderLidar2D
from mmwave_model_integrator.transforms.coordinate_transforms import cartesian_to_spherical,polar_to_cartesian

class RaadarHDGTEncoder(_GTEncoderLidar2D):

    def __init__(
            self,
            max_range_m:float = 10.8,
            num_range_bins:int = 256,
            angle_range_rad:list=[np.deg2rad(-90),np.deg2rad(90)],
            num_angle_bins:int = 512,
            x_max:float = 10,
            y_max:float = 10,
            z_min:float = -0.2, #radarHD originally -0.3
            z_max:float = 0.3,
            num_previous_frames=0
    ):
        
        self.max_range_m:float = max_range_m
        self.num_range_bins:int = num_range_bins
        
        self.angle_range_rad:list = angle_range_rad
        self.num_angle_bins:int = num_angle_bins

        self.num_previous_frames:int = num_previous_frames

        self.x_max:float = x_max
        self.y_max:float = y_max
        self.z_min:float = z_min
        self.z_max:float = z_max

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

        #filter in cartesian
        pc = self._filter_in_cartesian(lidar_pc)

        #convert points to spherical
        pc = cartesian_to_spherical(pc)

        #convert the points to a quantized grid
        grid = self.points_polar_to_grid(pc[:,])

        #perform BCC to remove miscellaneous smaller detections
        grid = self._apply_binary_connected_component_analysis_to_grid(grid)

        return grid
    
    ####################################################################
    #RadarHD Point Cloud Processing helper functions
    ####################################################################
    def _filter_in_cartesian(self, lidar_pc:np.ndarray) ->np.ndarray:

        mask = (lidar_pc[:,0] >= 0) * (lidar_pc[:,0] <= self.x_max) & \
                (lidar_pc[:,1] >= -self.y_max) * (lidar_pc[:,0] <= self.y_max) & \
                (lidar_pc[:,2] >= self.z_min) * (lidar_pc[:,2] <= self.z_max)
        
        return lidar_pc[mask,0:3]


