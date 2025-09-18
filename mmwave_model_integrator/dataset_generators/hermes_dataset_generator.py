import numpy as np

from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_model_integrator.dataset_generators._offline_dataset_generator import _OfflineDatasetGenerator
from mmwave_model_integrator.ground_truth_encoders.hermes_gt_encoder import HermesGTEncoder
from mmwave_model_integrator.input_encoders.hermes_encoder import HermesEncoder


class HermesDatasetGenerator(_OfflineDatasetGenerator):

    def __init__(
            self, 
            generated_dataset_path: str, 
            dataset_handler: CpslDS, 
            input_encoder: HermesEncoder, 
            ground_truth_encoder: HermesGTEncoder, 
            generated_file_name: str = "frame", 
            input_encoding_folder: str = "x_s", 
            ground_truth_encoding_folder: str = "y_s", 
            clear_existing_data: bool = False):

        self.input_encoder: HermesEncoder = None

        super().__init__(
            generated_dataset_path=generated_dataset_path,
            dataset_handler=dataset_handler,
            input_encoder=input_encoder,
            ground_truth_encoder=ground_truth_encoder,
            generated_file_name=generated_file_name,
            input_encoding_folder=input_encoding_folder,
            ground_truth_encoding_folder=ground_truth_encoding_folder,
            clear_existing_data=clear_existing_data)
    
    ####################################################################
    #Methods to implemented for RadSAR implementation
    ####################################################################

    def _get_input_encoding_from_dataset(self,idx:int)->np.ndarray:
        """method implemented by child class to access the dataset
        and encode an input sample for model training

        Args:
            idx (int): sample index in the cpsl_dataset of the radar data

        Returns:
            np.ndarray: endoded input into the model
        """

        adc_cube = self.dataset_handler.get_radar_data(idx)

        try: #try accessing the full odometry data
            vel_data = np.mean(self.dataset_handler.get_vehicle_odom_data(idx)[:,8:11],axis=0)
        except AssertionError: #if not just get the x velocity (forward)
            vel = np.mean(self.dataset_handler.get_vehicle_vel_data(idx)[:,1])
            vel_data = np.array([vel,0,0])
        
        return self.input_encoder.encode(
            adc_data_cube=adc_cube,
            vels=vel_data
        )
    
    def _get_output_encoding_from_dataset(self,idx:int)->np.ndarray:
        """method implemented by child class to access the dataset
        and encode an output sample for model training

        Args:
            idx (int): sample index in the cpsl_dataset to get output encoding data from

        Returns:
            np.ndarray: encoded lidar PC grid for training
        """
        
        lidar_pc = self.dataset_handler.get_lidar_point_cloud_raw(idx)

        if self.input_encoder.full_encoding_ready:
            return self.ground_truth_encoder.encode(
                lidar_pc=lidar_pc,
                input_encoding=self.input_encoder.encoded_data
            )
        else:
            return np.empty(shape=(0))
    
