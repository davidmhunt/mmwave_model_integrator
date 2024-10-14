import numpy as np
from scipy.spatial.distance import cdist

from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_model_integrator.plotting.plotter_rng_az_to_pc import PlotterRngAzToPC
from mmwave_model_integrator.encoders._radar_range_az_encoder import _RadarRangeAzEncoder
from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner

class AnalyzerRngAzToPC:

    def __init__(self,
                 cpsl_dataset:CpslDS,
                 range_az_encoder:_RadarRangeAzEncoder,
                 model_runner:_ModelRunner,
                 lidar_pc_polar_decoder:_lidarPCPolarDecoder,
                 temp_dir_path="~/Downloads/odometry_temp") -> None:
        
        self.dataset:CpslDS = cpsl_dataset
        self.range_az_encoder:_RadarRangeAzEncoder = range_az_encoder
        self.model_runner:_ModelRunner = model_runner
        self.lidar_pc_polar_decoder:_lidarPCPolarDecoder = lidar_pc_polar_decoder
        
        self.temp_dir_path = temp_dir_path
        self.temp_file_name = "frame"

        self.next_frame:int = 0

        self.reset()

    ####################################################################
    #Helper functions - Analyzing performance
    ####################################################################

    def reset(self):
        pass

    ####################################################################
    #Computing Distance Metrics
    ####################################################################

    def _compute_distance_metrics(self,sample_idx, print_result = False):
        """Returns the chamfer and hausdorff distances between the points in the ground truth point cloud and predicted point cloud

        Args:
            sample_idx (int): The sample index of the point cloud to compute distances for
            print_result (bool, optional): On True, prints the distances. Defaults to False.

        Returns:
            double,double,double,double: Chamfer distance (m), Hausdorff distance (m), Chamfer (radCloud) distance (m), Modified hausdorff distance (radCloud) distance
        """

        try: 
            distances = self._compute_euclidian_distances(sample_idx)

            #compute alternative metrics
            chamfer_alt = self._compute_chamfer_alt(distances)
            hausdorff_alt = self._compute_hausdorff_alt(distances)

            #compute RadCloud Metrics
            chamfer_radCloud = self._compute_chamfer_radCloud(distances)
            modified_hausdorff_radCloud = self._compute_modified_hausdorff_radCloud(distances)

            if print_result:
                print("Chamfer: {}, Hausdorff: {}".format(chamfer_alt,hausdorff_alt))

            return chamfer_alt,hausdorff_alt, chamfer_radCloud, modified_hausdorff_radCloud
        except ValueError:
            self.num_failed_predictions += 1
            return 0,0,0,0
    
    def _compute_euclidian_distances(self, sample_idx):
        """Compute the euclidian distance between all of the points in the ground truth point cloud and the predicted point cloud

        Args:
            sample_idx (int): The sample index of the point cloud to compare

        Returns:
            ndarray: an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud
        """
        adc_cube = self.dataset.get_radar_data(idx=sample_idx)

        #get the ground truth grid, convert to spherical points, convert to cartesian points
        rng_az_resp = self.range_az_encoder.encode(adc_cube)
        pred = self.model_runner.make_prediction(input=rng_az_resp)
        pc_pred = self.lidar_pc_polar_decoder.convert_polar_to_cartesian(
            self.lidar_pc_polar_decoder.decode(pred)
        )

        #get the prediction, convert to spherical points, convert to cartesian points
        #TODO: Get the output ground truth encoding

        return cdist(ground_truth,pc_pred,metric="euclidean")
    
    def _compute_hausdorff_alt(self,distances):
        """ (Alternative to radCloud hausdorff distance)
            Compute the Hausdorff distance between the predicted point cloud and the ground truth point cloud
            Note: formula from: https://pdal.io/en/latest/apps/hausdorff.html
        Args:
            distances (ndarray): an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud

        Returns:
            double: hausdorff distance
        """

        ground_truth_mins = np.min(distances,axis=1)
        prediction_mins = np.min(distances,axis=0)

        return  np.max([np.max(ground_truth_mins),np.max(prediction_mins)])
    
    def _compute_modified_hausdorff_radCloud(self,distances):
        """Compute the Hausdorff distance between the predicted point cloud and the ground truth point cloud
            Note: formula from: https://github.com/akarsh-prabhakara/RadarHD/blob/main/eval/pc_distance.m
        Args:
            distances (ndarray): an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud

        Returns:
            double: hausdorff distance from radCloud
        """

        ground_truth_mins = np.min(distances,axis=1)
        prediction_mins = np.min(distances,axis=0)

        return np.max([np.median(ground_truth_mins),np.median(prediction_mins)])
    
    def _compute_chamfer_alt(self,distances):
        """(Alternative to radcloud chamfer distance)
            Compute the Chamfer distance between the predicted point cloud and the ground truth point cloud
            Note: formula from: https://github.com/DavidWatkins/chamfer_distance
        Args:
            distances (ndarray): an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud

        Returns:
            double: Chamfer distance
        """

        ground_truth_mins = np.min(distances,axis=1)
        prediction_mins = np.min(distances,axis=0)

        #square the distances
        ground_truth_mins = np.square(ground_truth_mins)
        prediction_mins = np.square(prediction_mins)

        return np.mean(ground_truth_mins) + np.mean(prediction_mins)
    
    def _compute_chamfer_radCloud(self,distances):
        """Compute the Chamfer distance between the predicted point cloud and the ground truth point cloud as used in radCloud
            Note: formula from: https://github.com/akarsh-prabhakara/RadarHD/blob/main/eval/pc_distance.m
        Args:
            distances (ndarray): an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud

        Returns:
            double: Chamfer distance from radCloud
        """

        ground_truth_mins = np.min(distances,axis=1)
        prediction_mins = np.min(distances,axis=0)

        return (0.5 * np.mean(ground_truth_mins)) + (0.5 * np.mean(prediction_mins))