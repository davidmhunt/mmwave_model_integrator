import numpy as np
import cv2

from mmwave_model_integrator.transforms.coordinate_transforms import cartesian_to_spherical,spherical_to_cartesian
from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder

class _GTEncoderLidar2D(_GTEncoder):
    """Encoder specifically designed to encode lidar data into
    the format used to train a model
    """

    def __init__(self) -> None:

        #flag to note whether a full encoding is ready or not
        #(for encoders that encode a series of frames)
        self.full_encoding_ready = False
        

        #range and angle bins - SET BY CHILD CLASS
        if not self.num_angle_bins:
            self.num_angle_bins = None
        self.range_bins_m:np.ndarray = None
        self.angle_bins_rad:np.ndarray = None

        #array for the encoded data
        #NOTE: #indexed/implemented depending on child class
        self.encoded_data:np.ndarray = None
        
        #complete the configuration
        super().__init__()

        return

    def encode(self,lidar_pc:np.ndarray)->np.ndarray:
        """Implemented by child class to encode data for a specific
        model

        Args:
            lidar_pc (np.ndarray): N x 3 3D point cloud of lidar data

        Returns:
            np.ndarray: np.ndarray consisting of data to be output
                from the model
        """
        pass
    
    ####################################################################
    #Grid processing helper functions
    ####################################################################
    
    def grid_to_polar_points(self,grid:np.ndarray)->np.ndarray:
        """Convert a quantized grid to polar coordinates

        Args:
            grid (np.ndarray): rng_bins x az_bins NP array where
                nonzero values indicate occupancy in that area

        Returns:
            np.ndarray: Nx2 array of points in polar coordinates
        """
        #get the nonzero coordinates
        rng_idx,az_idx = np.nonzero(grid)

        rng_vals = self.range_bins_m[rng_idx]
        az_vals = self.angle_bins_rad[az_idx]

        return np.column_stack((rng_vals,az_vals))
    
    def points_polar_to_grid(self,points_polar:np.ndarray)->np.ndarray:
        """Convert a set of points to a quantized polar grid

        Args:
            points (np.ndarray): Nx2 or Nx3 NP array of points in 
                polar (or spherical) coordinates to quantize

        Returns:
            np.ndarray: rng_bins x az_bins NP array where
                nonzero values indicate occupancy in that area
        """

        #define the out grid
        out_grid = np.zeros((
            self.range_bins_m.shape[0],
            self.angle_bins_rad.shape[0]))

        #identify the nearest point from the pointcloud
        r_idx = np.argmin(np.abs(self.range_bins_m - points_polar[:,0][:,None]),axis=1)
        az_idx = np.argmin(np.abs(self.angle_bins_rad - points_polar[:,1][:,None]),axis=1)

        out_grid[r_idx,az_idx] = 1

        return out_grid