import numpy as np
import cv2

from mmwave_model_integrator.transforms.coordinate_transforms import cartesian_to_spherical,spherical_to_cartesian

class _Lidar2DPCEncoder:
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
        self.configure()

        return

    def configure(self):
        """Configure the point cloud encoder. Remaining functionality
        must be implemented by child class to configure its modules.
        """

        return

    def encode(self,lidar_pc:np.ndarray)->np.ndarray:
        """Implemented by child class to encode data for a specific
        model

        Args:
            lidar_pc (np.ndarray): N x 3 3D point cloud of lidar data

        Returns:
            np.ndarray: np.ndarray consisting of data to be input
                into the model
        """
        pass

    ####################################################################
    #Point Cloud Processing helper functions
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