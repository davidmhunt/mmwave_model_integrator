import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_model_integrator.plotting.plotter_rng_az_to_pc import PlotterRngAzToPC
from mmwave_model_integrator.input_encoders._radar_range_az_encoder import _RadarRangeAzEncoder
from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.output_encoders._lidar_2D_pc_encoder import _Lidar2DPCEncoder
from mmwave_model_integrator.transforms.coordinate_transforms import polar_to_cartesian

class AnalyzerRngAzToPC:

    def __init__(self,
                 cpsl_dataset:CpslDS,
                 input_encoder:_RadarRangeAzEncoder,
                 model_runner:_ModelRunner,
                 prediction_decoder:_lidarPCPolarDecoder,
                 ground_truth_encoder: _Lidar2DPCEncoder,
                 temp_dir_path="~/Downloads/odometry_temp") -> None:
        
        self.dataset:CpslDS = cpsl_dataset
        self.input_encoder:_RadarRangeAzEncoder = input_encoder
        self.model_runner:_ModelRunner = model_runner
        self.prediction_decoder:_lidarPCPolarDecoder = prediction_decoder
        self.ground_truth_encoder:_Lidar2DPCEncoder = ground_truth_encoder
        
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
    #Computing distance metrics across an entire dataset
    ####################################################################
    
    def compute_all_distance_metrics(self, save_to_file=False,file_name = "trained"):

        #initialize arrays to store the distributions in (alternative metrics)
        chamfer_distances = np.zeros((self.dataset.num_frames))
        hausdorff_distances = np.zeros((self.dataset.num_frames))

        #initialize arrays to store the distributions in (radCloud metrics)
        chamfer_distances_radarHD = np.zeros((self.dataset.num_frames))
        hausdorff_distances_radarHD = np.zeros((self.dataset.num_frames))

        #reset failed sample tracking
        self.num_failed_predictions = 0

        #compute the distances for each of the arrays
        print("Analyzer.compute_all_distance_metrics: Computing distance metrics")
        for i in tqdm(range(self.dataset.num_frames)):
            chamfer_distances[i],hausdorff_distances[i],chamfer_distances_radarHD[i],hausdorff_distances_radarHD[i] = \
                self._compute_distance_metrics(sample_idx=i,print_result=False)
        
        print("Analyzer.compute_all_distance_metrics: number failed predictoins {} of {} ({}%)".format(
            self.num_failed_predictions,
            self.dataset.num_frames,
            float(self.num_failed_predictions) / float(self.dataset.num_frames)
        ))
        
        #save metrics to .npy file if set to true
        if save_to_file:
            
            #save chamfer
            name = "{}_chamfer.npy".format(file_name)
            np.save(name,chamfer_distances)

            #save hausdroff
            name = "{}_hausdorf.npy".format(file_name)
            np.save(name,hausdorff_distances)

            #save chamfer
            name = "{}_chamfer_radarHD.npy".format(file_name)
            np.save(name,chamfer_distances_radarHD)

            #save hausdroff
            name = "{}_hausdorff_radarHD.npy".format(file_name)
            np.save(name,hausdorff_distances_radarHD)

        return chamfer_distances,hausdorff_distances, chamfer_distances_radarHD, hausdorff_distances_radarHD
        
    ####################################################################
    #Computing Distance Metrics (for a given sample index)
    ####################################################################

    def _compute_distance_metrics(self,sample_idx, print_result = False):
        """Returns the chamfer and hausdorff distances between the points in the ground truth point cloud and predicted point cloud

        Args:
            sample_idx (int): The sample index of the point cloud to compute distances for
            print_result (bool, optional): On True, prints the distances. Defaults to False.

        Returns:
            double,double,double,double: Chamfer distance (m), Hausdorff distance (m), Chamfer (radarHD) distance (m), Modified hausdorff distance (radarHD) distance
        """

        try: 
            distances = self._compute_euclidian_distances(sample_idx)

            #compute alternative metrics
            chamfer = self._compute_chamfer(distances)
            hausdorff = self._compute_hausdorff(distances)

            #compute RadCloud Metrics
            chamfer_radarHD = self._compute_chamfer_radarHD(distances)
            hausdorff_radarHD = self._compute_hausdorff_radarHD(distances)

            if print_result:
                print("Chamfer: {}, Hausdorff: {}".format(chamfer,hausdorff))

            return chamfer,hausdorff, chamfer_radarHD, hausdorff_radarHD
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
        rng_az_resp = self.input_encoder.encode(adc_cube)
        pred = self.model_runner.make_prediction(input=rng_az_resp)
        pc_pred = self.prediction_decoder.convert_polar_to_cartesian(
            self.prediction_decoder.decode(pred)
        )

        #compute the ground truth point cloud
        lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=0)
        grid = self.ground_truth_encoder.encode(lidar_pc)
        quantized_pc = self.ground_truth_encoder.grid_to_polar_points(grid)
        ground_truth = polar_to_cartesian(quantized_pc)

        #get the prediction, convert to spherical points, convert to cartesian points
        #TODO: Get the output ground truth encoding

        return cdist(ground_truth,pc_pred,metric="euclidean")
    
    def _compute_hausdorff(self,distances):
        """ (hausdorff distance)
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
    
    def _compute_hausdorff_radarHD(self,distances):
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
    
    def _compute_chamfer(self,distances):
        """(chamfer distance)
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
    
    def _compute_chamfer_radarHD(self,distances):
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