import tqdm
import numpy as np

from cpsl_datasets.cpsl_ds import CpslDS

from mmwave_model_integrator.plotting.plotter_rng_az_to_pc import PlotterRngAzToPC
from mmwave_model_integrator.input_encoders._radar_range_az_encoder import _RadarRangeAzEncoder
from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders._gt_encoder_lidar2D import _GTEncoderLidar2D
from mmwave_model_integrator.plotting._movie_generator import _MovieGenerator

class MovieGeneratorRngAzToPC(_MovieGenerator):

   
    def __init__(self,
                 cpsl_dataset:CpslDS,
                 plotter:PlotterRngAzToPC,
                 input_encoder:_RadarRangeAzEncoder,
                 runner:_ModelRunner=None,
                 decoder:_lidarPCPolarDecoder=None,
                 ground_truth_encoder:_GTEncoderLidar2D=None,
                 temp_dir_path="~/Downloads/odometry_temp",
                 ) -> None:
        
        self.dataset:CpslDS = cpsl_dataset

        super().__init__(
            plotter=plotter,
            input_encoder=input_encoder,
            runner=runner,
            decoder=decoder,
            ground_truth_encoder=ground_truth_encoder,
            temp_dir_path=temp_dir_path
        )
    
    ####################################################################
    #Helper functions - movie generation
    #################################################################### 

    def reset(self):
        """Reset the movie generator
        """
        super().reset()

    
    def generate_movie_frames(self):

        #prime the dataset to ensure that the encoders and decoders have sufficient data 
        #with an encoding ready to go
        start_idx=0
        self.input_encoder.reset()

        if self.ground_truth_encoder:
            while (not self.input_encoder.full_encoding_ready) or \
                (not self.ground_truth_encoder.full_encoding_ready):

                #get the radar data
                adc_cube = self.dataset.get_radar_data(idx=start_idx)
                encoded_data = self.input_encoder.encode(adc_cube)

                lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=start_idx)
                grid = self.ground_truth_encoder.encode(lidar_pc)

                start_idx += 1
        else:
            while (not self.input_encoder.full_encoding_ready):

                #get the radar data
                adc_cube = self.dataset.get_radar_data(idx=start_idx)
                encoded_data = self.input_encoder.encode(adc_cube)

                start_idx += 1
        
        for i in tqdm.tqdm(range(start_idx,self.dataset.num_frames)):

            #get the adc cube
            adc_cube = self.dataset.get_radar_data(idx=i)

            if self.ground_truth_encoder:
                lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=i)
            else:
                lidar_pc = np.empty(0)
            
            self.plotter.plot_compilation(
                input_data=adc_cube,
                input_encoder=self.input_encoder,
                runner=self.runner,
                decoder=self.decoder,
                gt_data=lidar_pc,
                ground_truth_encoder=self.ground_truth_encoder,
                axs=self.axs,
                show=False
            )

            #save the frame
            self.save_frame(clear_axs=True)
    