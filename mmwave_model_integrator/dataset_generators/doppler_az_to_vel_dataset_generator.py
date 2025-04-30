import numpy as np

from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_model_integrator.dataset_generators._offline_dataset_generator import _OfflineDatasetGenerator
from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder
from mmwave_model_integrator.input_encoders._input_encoder import _InputEncoder


class DopplerAzToVelDatasetGenerator(_OfflineDatasetGenerator):

    def __init__(
            self, 
            generated_dataset_path: str, 
            dataset_handler: CpslDS, 
            input_encoder: _InputEncoder, 
            ground_truth_encoder: _GTEncoder, 
            generated_file_name: str = "frame", 
            input_encoding_folder: str = "doppler_az_resps", 
            ground_truth_encoding_folder: str = "vels", 
            clear_existing_data: bool = False):
        
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
    #Methods to implemented for Range Az -> PC Models
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
        return self.input_encoder.encode(adc_cube)
    
    def _get_output_encoding_from_dataset(self,idx:int)->np.ndarray:
        """method implemented by child class to access the dataset
        and encode an output sample for model training

        Args:
            idx (int): sample index in the cpsl_dataset to get output encoding data from

        Returns:
            np.ndarray: encoded velocity info for training
        """
        
        odom_data = self.dataset_handler.get_vehicle_odom_data(idx)
        return self.ground_truth_encoder.encode(odom_data)
    
