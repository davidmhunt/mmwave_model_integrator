import numpy as np
import cv2

from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder
from mmwave_model_integrator.input_encoders.hermes_encoder import HermesEncoder

class HermesGTEncoder(_GTEncoder):

    def __init__(
            self,
            hermes_encoder:HermesEncoder,
            candidate_points_threshold:float=0.25,
            valid_points_distance_threshold_m:float=0.65
    ):
        
        self.hermes_encoder = hermes_encoder
        self.candidate_points_threshold = candidate_points_threshold
        self.valid_points_distance_threshold_m = valid_points_distance_threshold_m

        #keeping track of the mesh grids
        self.x_s = self.hermes_encoder.interp_x_s
        self.y_s = self.hermes_encoder.interp_y_s

        self.gt_encoding:np.ndarray = None
        self.original_gt_encoding:np.ndarray = None

        super().__init__()
    
    def configure(self):

        return



    def encode(self, lidar_pc: np.ndarray, input_encoding: np.ndarray) -> np.ndarray:
        """Encodes data for the radcloud model

        Args:
            lidar_pc (np.ndarray): N x 3 3D point cloud of lidar data

        Returns:
            np.ndarray: Quantized grid as the ground truth output of the model
        """

        #remove the ground plane
        pc = self._filter_z_coordinates(lidar_pc)

        self.original_gt_encoding = self.points_to_grid(pc)
        original_gt_encoding_pc = self.grid_to_points(self.original_gt_encoding)
        
        #get the possible radar detections
        possible_detections = (input_encoding >= self.candidate_points_threshold).astype(np.uint8)
        possible_detections_pc = self.grid_to_points(possible_detections)

        filtered_possible_detection_pc = []

        #getting valid points from radar
        for point in possible_detections_pc:
            distances = np.linalg.norm(original_gt_encoding_pc - point, axis=1)
            if np.any(distances <= self.valid_points_distance_threshold_m):
                filtered_possible_detection_pc.append(point)
        
        # for point in original_gt_encoding_pc:
        #     distances = np.linalg.norm(possible_detections_pc - point, axis=1)
        #     if np.any(distances <= self.valid_points_distance_threshold_m):
        #         filtered_possible_detection_pc.append(point)

        #option 3
        # for point in possible_detections_pc:
        #     distances = np.linalg.norm(original_gt_encoding_pc - point, axis=1)
        #     within_threshold_idxs = np.where(distances <= self.valid_points_distance_threshold_m)[0]
        #     if within_threshold_idxs.size > 0:
        #     # Append the lidar point that is within the threshold distance
        #         filtered_possible_detection_pc.append(original_gt_encoding_pc[within_threshold_idxs[0]])


        filtered_possible_detection_pc = np.array(filtered_possible_detection_pc)

        #convert to a grid
        self.gt_encoding = self.points_to_grid(filtered_possible_detection_pc)

        #perform BCC to remove miscellaneous smaller detections
        # self.gt_encoding = self._apply_binary_connected_component_analysis_to_grid(self.gt_encoding)

        #specify full encoding ready
        self.full_encoding_ready = True

        return self.gt_encoding
    
    ####################################################################
    #Point Cloud Processing helper functions
    ####################################################################
    
    def _filter_z_coordinates(self,lidar_pc_raw:np.ndarray)->np.ndarray:

        #filter out the ground detections
        valid_idxs = (lidar_pc_raw[:,2] > -0.25) & \
                    (lidar_pc_raw[:,2] < 0.5)
        return lidar_pc_raw[valid_idxs,:]
    
    def points_to_grid(self,pc:np.ndarray)->np.ndarray:
        """Convert a cartesian point cloud to a quantized grid

        Args:
            pc (np.ndarray): Nx3 array of points in cartesian coordinates

        Returns:
            np.ndarray: rng_bins x az_bins NP array where
                nonzero values indicate occupancy in that area
        """

        # Extract x, y coordinates from the lidar point cloud
        x_coords = pc[:, 0]
        y_coords = pc[:, 1]

        # Find the nearest x and y bins for each point
        x_idx = np.argmin(np.abs(self.x_s[:,0] - x_coords[:, None]), axis=1)
        y_idx = np.argmin(np.abs(self.y_s[0,:] - y_coords[:, None]), axis=1)

        # Create the grid and mark the corresponding cells as occupied
        grid = np.zeros_like(self.x_s, dtype=np.uint8)
        grid[x_idx, y_idx] = 1

        
        return grid
    
    def grid_to_points(self,grid:np.ndarray)->np.ndarray:
        """Convert a quantized grid to cartesian coordinates

        Args:
            grid (np.ndarray): rng_bins x az_bins NP array where
                nonzero values indicate occupancy in that area

        Returns:
            np.ndarray: Nx2 array of points in cartesian coordinates
        """
        #get the nonzero coordinates
        x_idx,y_idx = np.nonzero(grid)

        x_vals = self.x_s[x_idx,0]
        y_vals = self.y_s[0,y_idx]

        return np.column_stack((x_vals,y_vals))
    

    def _apply_binary_connected_component_analysis_to_grid(self,grid:np.ndarray):
        
        # Perform connected component analysis
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(grid.astype(np.uint8))

        # Filter out isolated pixels
        min_size = 10 #min area in pixels

        #min height or width
        min_height = 3 #3
        min_width = 3 #3
        filtered_grid = np.zeros_like(grid)
        for i in range(1, num_labels):
            if ((min_size <= stats[i, cv2.CC_STAT_AREA]) and
            ((min_height <= stats[i, cv2.CC_STAT_HEIGHT]) or 
             min_width <= stats[i,cv2.CC_STAT_WIDTH])):
                filtered_grid[labels == i] = 1
        
        return filtered_grid