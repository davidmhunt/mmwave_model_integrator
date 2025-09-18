import numpy as np
import cv2

from mmwave_model_integrator.ground_truth_encoders._gt_encoder_lidar2D_polar import _GTEncoderLidar2DPolar
from mmwave_model_integrator.transforms.coordinate_transforms import cartesian_to_spherical,polar_to_cartesian

class RadCloudGTEncoder(_GTEncoderLidar2DPolar):

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

        #specify full encoding ready
        self.full_encoding_ready = True

        return grid
    
    ####################################################################
    #RadCloud Point Cloud Processing helper functions
    ####################################################################
    
    def _remove_ground_plane(self,points:np.ndarray)->np.ndarray:

        valid_points = points[:,2] > -0.2 #filter out ground
        valid_points = valid_points & (points[:,2] < 0.1) #higher elevation points


        return points[valid_points,:]
    
    def _filter_ranges_and_azimuths(self,points_spherical:np.ndarray):

        """Filter values in a point cloud (spherical coordinates) that are within the configured maximum range 
        and specified azimuth range

        Args:
            points_spherical (np.ndarray): Nx3 array of points in spherical coordinates
        """

        mask = (points_spherical[:,0] < self.max_range_m) & \
                (points_spherical[:,1] < self.angle_range_rad[0]) &\
                (points_spherical[:,1] > self.angle_range_rad[1])

        #filter out points not in radar's elevation beamwidth
        mask = mask & (np.abs(points_spherical[:,2] - np.pi/2) < 0.26) #was 0.26

        return points_spherical[mask]
    

    def _apply_binary_connected_component_analysis_to_grid(self,grid:np.ndarray):
        
        # Perform connected component analysis
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(grid.astype(np.uint8))

        # Filter out isolated pixels
        min_size = 4 #min area in pixels

        #min height or width
        min_height = 3
        min_width = 3
        filtered_grid = np.zeros_like(grid)
        for i in range(1, num_labels):
            if ((min_size <= stats[i, cv2.CC_STAT_AREA]) and
            ((min_height <= stats[i, cv2.CC_STAT_HEIGHT]) or 
             min_width <= stats[i,cv2.CC_STAT_WIDTH])):
                filtered_grid[labels == i] = 1
        
        return filtered_grid