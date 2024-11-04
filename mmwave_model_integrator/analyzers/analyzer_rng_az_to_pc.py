import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd
from IPython.display import display


from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_model_integrator.plotting.plotter_rng_az_to_pc import PlotterRngAzToPC
from mmwave_model_integrator.input_encoders._radar_range_az_encoder import _RadarRangeAzEncoder
from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders._gt_encoder_lidar2D import _GTEncoderLidar2D
from mmwave_model_integrator.transforms.coordinate_transforms import polar_to_cartesian

class AnalyzerRngAzToPC:

    def __init__(self,
                 cpsl_dataset:CpslDS,
                 input_encoder:_RadarRangeAzEncoder,
                 model_runner:_ModelRunner,
                 prediction_decoder:_lidarPCPolarDecoder,
                 ground_truth_encoder: _GTEncoderLidar2D,
                 temp_dir_path="~/Downloads/odometry_temp") -> None:
        
        self.dataset:CpslDS = cpsl_dataset
        self.input_encoder:_RadarRangeAzEncoder = input_encoder
        self.model_runner:_ModelRunner = model_runner
        self.prediction_decoder:_lidarPCPolarDecoder = prediction_decoder
        self.ground_truth_encoder:_GTEncoderLidar2D = ground_truth_encoder
        
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
    #Compute and show summary statistics
    ####################################################################
    def show_all_summary_statistics(self,
                                chamfer_distances:np.ndarray=np.empty(0),
                                hausdorff_distances:np.ndarray=np.empty(0),
                                chamfer_distances_radarHD:np.ndarray=np.empty(0),
                                modified_hausdorff_distances_radarHD:np.ndarray=np.empty(0)):
        """Display a set of summary statistics in a table

        Args:
            chamfer_distances (np.ndarray): _description_
            hausdorff_distances (np.ndarray): _description_
            chamfer_distances_radarHD (np.ndarray, optional): _description_. Defaults to None.
            modified_hausdorff_distances_radarHD (np.ndarray, optional): _description_. Defaults to None.
        """
        #compute stats for hausdorff
        hausdorff_mean = np.mean(hausdorff_distances)
        hausdorff_median = np.median(hausdorff_distances)
        hausdorff_tail_90_percent = self._get_percentile(hausdorff_distances,0.90)
        
        #compute stats for modified hausdorff (radarHD)
        hausdorff_radar_HD_mean = \
            np.mean(modified_hausdorff_distances_radarHD)
        hausdorff_radar_HD_median = \
            np.median(modified_hausdorff_distances_radarHD)
        hausdorff_radar_HD_tail_90_percent = \
            self._get_percentile(modified_hausdorff_distances_radarHD, 0.90)

        #compute stats for chamfer
        chamfer_mean = np.mean(chamfer_distances)
        chamfer_median = np.median(chamfer_distances)
        chamfer_tail_90_percent = self._get_percentile(chamfer_distances,0.90)
        
        #compute stats for chamfer (RadarHD)
        chamfer_radarHD_mean = \
            np.mean(chamfer_distances_radarHD)
        chamfer_radarHD_median = \
            np.median(chamfer_distances_radarHD)
        chamfer_radarHD_tail_90_percent = \
            self._get_percentile(chamfer_distances_radarHD, 0.90)
        
        #generate and display table
        dict = {
            'Metric': ["Mean","Median","90th percentile"],
            'Hausdorff':[
                hausdorff_mean,
                hausdorff_median,
                hausdorff_tail_90_percent],
            'Modified Hausdorff (radarHD)':[
                hausdorff_radar_HD_mean,
                hausdorff_radar_HD_median,
                hausdorff_radar_HD_tail_90_percent],
            'Chamfer':[
                chamfer_mean,
                chamfer_median,
                chamfer_tail_90_percent],
            'Chamfer (radarHD)':[
                chamfer_radarHD_mean,
                chamfer_radarHD_median,
                chamfer_radarHD_tail_90_percent]
        }

        df = pd.DataFrame(dict)
        display(df)

    def _get_percentile(self,distances:np.ndarray, percentile:float):

        sorted_data = np.sort(distances)
        p = 1. * np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        #compute the index of the percentile
        idx = (np.abs(p - percentile)).argmin()

        return sorted_data[idx]
    ####################################################################
    #Computing distance metrics across an entire dataset
    ####################################################################
    
    def compute_all_distance_metrics(self, save_to_file=False,file_name = "trained"):

        #prime the dataset to ensure that the encoders and decoders have sufficient data 
        #with an encoding ready to go
        start_idx=0
        self.input_encoder.reset_history()

        while (not self.input_encoder.full_encoding_ready) or \
            (not self.ground_truth_encoder.full_encoding_ready):

            #get the radar data
            adc_cube = self.dataset.get_radar_data(idx=start_idx)
            encoded_data = self.input_encoder.encode(adc_cube)

            lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=start_idx)
            grid = self.ground_truth_encoder.encode(lidar_pc)

            start_idx += 1
        
        #initialize arrays to store the distributions in (alternative metrics)
        chamfer_distances = np.zeros((self.dataset.num_frames - start_idx))
        hausdorff_distances = np.zeros((self.dataset.num_frames - start_idx))

        #initialize arrays to store the distributions in (radCloud metrics)
        chamfer_distances_radarHD = np.zeros((self.dataset.num_frames - start_idx))
        hausdorff_distances_radarHD = np.zeros((self.dataset.num_frames - start_idx))

        #reset failed sample tracking
        self.num_failed_predictions = 0

        #compute the distances for each of the arrays
        print("Analyzer.compute_all_distance_metrics: Computing distance metrics")
        for sample_idx in tqdm(range(start_idx,self.dataset.num_frames)):

            save_idx = sample_idx - start_idx

            chamfer_distances[save_idx], \
            hausdorff_distances[save_idx], \
            chamfer_distances_radarHD[save_idx], \
            hausdorff_distances_radarHD[save_idx] = \
                self._compute_distance_metrics(sample_idx=sample_idx,print_result=False)
        
        print("Analyzer.compute_all_distance_metrics: number failed predictoins {} of {} ({}%)".format(
            self.num_failed_predictions,
            self.dataset.num_frames - start_idx,
            float(self.num_failed_predictions) / float(self.dataset.num_frames - start_idx)
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
        lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=sample_idx)
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