import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd
from IPython.display import display


from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_model_integrator.input_encoders.hermes_encoder import HermesEncoder
from mmwave_model_integrator.decoders.heremes_decoder import HermesDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders.hermes_gt_encoder import HermesGTEncoder
from mmwave_model_integrator.transforms.coordinate_transforms import polar_to_cartesian

from mmwave_model_integrator.analyzers.analyzer_rng_az_to_pc import AnalyzerRngAzToPC

class AnalyzerHermes(AnalyzerRngAzToPC):

    def __init__(self,
                 cpsl_dataset:CpslDS,
                 input_encoder:HermesEncoder,
                 model_runner:_ModelRunner,
                 prediction_decoder:HermesDecoder,
                 ground_truth_encoder: HermesEncoder,
                 temp_dir_path="~/Downloads/odometry_temp") -> None:

        self.input_encoder:HermesEncoder = None
        self.ground_truth_encoder:HermesGTEncoder = None

        super().__init__(
            cpsl_dataset=cpsl_dataset,
            input_encoder=input_encoder,
            model_runner=model_runner,
            prediction_decoder=prediction_decoder,
            ground_truth_encoder=ground_truth_encoder,
            temp_dir_path=temp_dir_path
        )

    ####################################################################
    #Computing distance metrics across an entire dataset
    ####################################################################
    
    def compute_all_distance_metrics(self, save_to_file=False,file_name = "trained"):

        #prime the dataset to ensure that the encoders and decoders have sufficient data 
        #with an encoding ready to go
        start_idx=0
        self.input_encoder.reset()

        while (not self.input_encoder.full_encoding_ready) or \
            (not self.ground_truth_encoder.full_encoding_ready):

            #get the radar data
            adc_cube = self.dataset.get_radar_data(idx=start_idx)

            try: #try accessing the full odometry data
                vel_data = np.mean(self.dataset.get_vehicle_odom_data(start_idx)[:,8:11],axis=0)
            except AssertionError: #if not just get the x velocity (forward)
                vel = np.mean(self.dataset.get_vehicle_vel_data(start_idx)[:,1])
                vel_data = np.array([vel,0,0])
            encoded_data = self.input_encoder.encode(
                adc_data_cube=adc_cube,
                vels=vel_data
            )

            if self.input_encoder.full_encoding_ready:
                lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=start_idx)
                grid = self.ground_truth_encoder.encode(
                    lidar_pc=lidar_pc,
                    input_encoding=encoded_data
                )

            start_idx += 1
        
        #initialize arrays to store the distributions in (alternative metrics)
        chamfer_distances = np.zeros((self.dataset.num_frames - start_idx))
        hausdorff_distances = np.zeros((self.dataset.num_frames - start_idx))

        #initialize arrays to store the distributions in (radCloud metrics)
        chamfer_distances_radarHD = np.zeros((self.dataset.num_frames - start_idx))
        hausdorff_distances_radarHD = np.zeros((self.dataset.num_frames - start_idx))

        #reset failed sample tracking
        self.num_invalid_encodings = 0

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
            self.num_invalid_encodings,
            self.dataset.num_frames - start_idx,
            float(self.num_invalid_encodings) / float(self.dataset.num_frames - start_idx)
        ))

        #remove all zero values (resulting from invalid encodings)
        chamfer_distances = chamfer_distances[chamfer_distances != 0]
        hausdorff_distances = hausdorff_distances[hausdorff_distances!=0]
        chamfer_distances_radarHD=chamfer_distances_radarHD[chamfer_distances_radarHD!=0]
        hausdorff_distances_radarHD=hausdorff_distances_radarHD[hausdorff_distances_radarHD!=0]
        
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

            if distances.shape[0] > 0:

                #compute alternative metrics
                chamfer = self._compute_chamfer(distances)
                hausdorff = self._compute_hausdorff(distances)

                #compute RadCloud Metrics
                chamfer_radarHD = self._compute_chamfer_radarHD(distances)
                hausdorff_radarHD = self._compute_hausdorff_radarHD(distances)

                if print_result:
                    print("Chamfer: {}, Hausdorff: {}".format(chamfer,hausdorff))

                return chamfer,hausdorff, chamfer_radarHD, hausdorff_radarHD
            else:
                return 0,0,0,0
            
        except ValueError:
            self.num_invalid_encodings += 1
            return 0,0,0,0
    
    def _compute_euclidian_distances(self, sample_idx):
        """Compute the euclidian distance between all of the points in the ground truth point cloud and the predicted point cloud

        Args:
            sample_idx (int): The sample index of the point cloud to compare

        Returns:
            ndarray: an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud
        """
        adc_cube = self.dataset.get_radar_data(idx=sample_idx)

        try: #try accessing the full odometry data
            vel_data = np.mean(self.dataset.get_vehicle_odom_data(sample_idx)[:,8:11],axis=0)
        except AssertionError: #if not just get the x velocity (forward)
            vel = np.mean(self.dataset.get_vehicle_vel_data(sample_idx)[:,1])
            vel_data = np.array([vel,0,0])

        #get the ground truth grid, convert to spherical points, convert to cartesian points
        input_encoding = self.input_encoder.encode(
            adc_data_cube=adc_cube,
            vels=vel_data
        )

        if self.input_encoder.full_encoding_ready:
            pred = self.model_runner.make_prediction(input=input_encoding)
            pc_pred = self.prediction_decoder.decode(pred)

            #compute the ground truth point cloud
            lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=sample_idx)
            ground_truth = self.ground_truth_encoder.encode(lidar_pc,input_encoding)

            #TODO: compared to the ground truth mask
            ground_truth = self.ground_truth_encoder.grid_to_points(ground_truth)

            #get the prediction, convert to spherical points, convert to cartesian points
            #TODO: Get the output ground truth encoding

            return cdist(ground_truth,pc_pred,metric="euclidean")
        else: #no full encoding ready
            return np.empty(shape=(0))