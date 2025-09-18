import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from mmwave_model_integrator.input_encoders._input_encoder import _InputEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder
from cpsl_datasets.cpsl_ds import CpslDS

class _OfflineDatasetGenerator:
    """Parent class for generating datasets to train models using the mmwave model 
    integrator pipeline by looking at a dataset directly (i.e.; in an offline method)
    """

    def __init__(
                self,
                generated_dataset_path:str,
                dataset_handler:CpslDS,
                input_encoder:_InputEncoder,
                ground_truth_encoder:_GTEncoder,
                generated_file_name:str = "frame",
                input_encoding_folder:str = "x_s",
                ground_truth_encoding_folder:str = "y_s",
                clear_existing_data:bool=False
                 ):
        """Initialize the dataset generator

        Args:
            input_encoder (_InputEncoder): _description_
            ground_truth_encoder (_GTEncoder, optional): _description_. Defaults to None.
        """

        #parameters for the generated dataset
        self.generated_dataset_path:str = generated_dataset_path
        self.generated_file_name:str = generated_file_name
        self.input_encoding_folder:str = input_encoding_folder
        self.ground_truth_encoding_folder:str = ground_truth_encoding_folder
        self.clear_existing_data:bool = False

        #tracking the current sample number
        self.current_sample_number = 0
        self.save_file_number_offset = 10000 #offset applied to save file names

        #encoders for input and output
        self.input_encoder:_InputEncoder = input_encoder
        self.ground_truth_encoder:_GTEncoder = ground_truth_encoder

        #dataset
        self.dataset_handler:CpslDS = dataset_handler
        
        #flag for clearing existing data in a dataset
        self.clear_existing_data = clear_existing_data

        self.config_generated_dataset_paths()
    
    ####################################################################
    #Configuring dataset paths and preparing generated folders
    ####################################################################

    def reset(self,generated_dataset_path:str):
        """Reset the 

        Args:
            generated_dataset_path (str): _description_
        """
        
        #specify the new generated dataset path
        self.generated_dataset_path:str = generated_dataset_path
        
        #reset the current sample index
        self.current_sample_number = 0

        #re-configure the dataset generator
        self.config_generated_dataset_paths()
    
    def config_generated_dataset_paths(self):
        """Configure the path to the generated dataset
        """     
        
        #check for the generated dataset folder
        self._check_for_directory(self.generated_dataset_path,
                                  clear_contents=self.clear_existing_data)

        
        #setup input encoding folder
        input_save_path = os.path.join(
                self.generated_dataset_path,
                self.input_encoding_folder)
        self._check_for_directory(
            path= input_save_path,
            clear_contents=self.clear_existing_data
        )
        if not self.clear_existing_data:
            self.current_sample_number = \
                self._get_highest_idx_from_directory(input_save_path) + 1
        
        #setup ground truth encoding folder
        ground_truth_save_path = os.path.join(
                self.generated_dataset_path,
                self.ground_truth_encoding_folder)
        self._check_for_directory(
            path= ground_truth_save_path,
            clear_contents=self.clear_existing_data
        )
        if not self.clear_existing_data:
            gt_idx = \
                self._get_highest_idx_from_directory(ground_truth_save_path) + 1
            
            if gt_idx != self.current_sample_number:
                raise RuntimeError(
                    "DatasetGenerator: number of existing GT files not equal to number of input files")

        return
    
    def _check_for_directory(self,path, clear_contents = False):
        """Checks to see if a directory exists, 
        if the directory does not exist, attepts to create the directory.
        If it does exist, optionally removes all files

        Args:
            path (str): path to the directory to create
            clear_contents (bool, optional): removes all contents in the directory on True. Defaults to False.
        """

        if os.path.isdir(path):
            print("DatasetGenerator._check_for_directory: found directory {}".format(path))

            if clear_contents:
                print("DatasetGenerator._check_for_directory: clearing contents of {}".format(path))

                #clear the contents
                for file in os.listdir(path):
                    file_path = os.path.join(path,file)

                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print("Failed to delete {}".format(path))
        else:
            print("DatasetGenerator._check_for_directory: creating directory {}".format(path))
            os.makedirs(path)
        return

    def _get_highest_idx_from_directory(self,path:str)->int:
        """For a directory with files inside, return the value of the largest
        file index in the directory

        Args:
            path (str): the path of the directory to evaluate

        Returns:
            int: -1 if directory is empty, sample_number if not empty
        """

        contents = sorted(os.listdir(path))

        if len(contents) > 0:
            last_file = contents[-1]

            #get the most recent sample number
            sample_number = last_file.split("_")[1]
            sample_number = int(sample_number.split(".")[0]) \
                - self.save_file_number_offset
        
            return sample_number
        else:
            return -1    
    
    ####################################################################
    #Generating datasets
    ####################################################################

    def generate_dataset_from_multiple_scenarios(self,scenario_paths:list):

        for i in range(len(scenario_paths)):

            print("Generating data from scenario {} of {}".format(i,len(scenario_paths)))
            self.generate_dataset(scenario_paths[i])
    
    def generate_dataset(self,scenario_path:str):
        """Generate a dataset from a given scenario path

        Args:
            scenario_path (str): the path to a cpsl dataset from a given scenario
        """
        #load the new dataset
        self.dataset_handler.load_new_dataset(scenario_path)

        #reset the encoders
        self.input_encoder.reset()
        self.ground_truth_encoder.reset()

        #confirm dataset has data
        assert self.dataset_handler.num_frames>0, \
            "DatasetGenerator: no data samples found for scenario at path {}".format(scenario_path)

        #prime the dataset to ensure that the encoders and decoders have sufficient data 
        #with an encoding ready to go
        start_idx=0

        while (not self.input_encoder.full_encoding_ready) or \
            (not self.ground_truth_encoder.full_encoding_ready):

            #get the input data
            input_encoding = self._get_input_encoding_from_dataset(idx=start_idx)

            #get the output data
            output_encoding = self._get_output_encoding_from_dataset(idx=start_idx)

            start_idx += 1

        #save the first input and output encoding
        self.save_encodings_to_file(input_encoding,output_encoding)
        
        #generate the dataset
        for sample_idx in tqdm(range(start_idx,self.dataset_handler.num_frames)):

            #get the input data
            input_encoding = self._get_input_encoding_from_dataset(idx=sample_idx)

            if self.input_encoder.full_encoding_ready:

                #get the output data
                output_encoding = self._get_output_encoding_from_dataset(idx=sample_idx)

                #save the encoding to a file
                self.save_encodings_to_file(input_encoding,output_encoding)
        
        print("generated dataset now has {} samples".format(self.current_sample_number))
    
    ####################################################################
    #Dataset generation helper functions
    ####################################################################

    def save_encodings_to_file(
            self,
            input_encoding:np.ndarray,
            ground_truth_encoding:np.ndarray):
        """Save the generated encodings to a file and increment the current sample number

        Args:
            input_encoding (np.ndarray): input encoding expressed as a numpy array
            ground_truth_encoding (np.ndarray): ground truth encoding expressed as 
                a numpy array
        """
        #determine the save file name
        save_file_name = "{}_{}.npy".format(
            self.generated_file_name,
            self.current_sample_number + self.save_file_number_offset
        )

        #save the input
        path = os.path.join(
                self.generated_dataset_path,
                self.input_encoding_folder,
                save_file_name)
        np.save(path,input_encoding)

        #save the output
        path = os.path.join(
                self.generated_dataset_path,
                self.ground_truth_encoding_folder,
                save_file_name)
        np.save(path,ground_truth_encoding)

        #increment the current sample number of the generated dataset
        self.current_sample_number += 1

        return
    
    ####################################################################
    #Methods to implement in child class
    ####################################################################

    def _get_input_encoding_from_dataset(self,idx:int)->np.ndarray:
        """method implemented by child class to access the dataset
        and encode an input sample for model training

        Args:
            idx (int): sample index in the cpsl_dataset to get input encoding data from

        Returns:
            np.ndarray: encoded data expressed as a numpy array
        """
        return np.empty(0)
    
    def _get_output_encoding_from_dataset(self,idx:int)->np.ndarray:
        """method implemented by child class to access the dataset
        and encode an output sample for model training

        Args:
            idx (int): sample index in the cpsl_dataset to get output encoding data from

        Returns:
            np.ndarray: encoded data expressed as a numpy array
        """
        return np.empty(0)
