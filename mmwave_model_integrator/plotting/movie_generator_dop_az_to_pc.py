import tqdm
import numpy as np

from cpsl_datasets.cpsl_ds import CpslDS


from mmwave_model_integrator.input_encoders._doppler_az_encoder import _DopplerAzEncoder
from mmwave_model_integrator.decoders._decoder import _Decoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders._gt_encoder_odom_to_vel import _GTEncoderOdomToVel
from mmwave_model_integrator.plotting.plotter_doppler_az_to_vel import PlotterDopplerAzToVel

from mmwave_model_integrator.plotting._movie_generator import _MovieGenerator

class MovieGeneratorRngAzToPC(_MovieGenerator):

   
    def __init__(self,
                 cpsl_dataset:CpslDS,
                 plotter:PlotterDopplerAzToVel,
                 input_encoder:_DopplerAzEncoder,
                 runner:_ModelRunner=None,
                 decoder:_Decoder=None,
                 ground_truth_encoder:_GTEncoderOdomToVel=None,
                 temp_dir_path="~/Downloads/dop_az_to_pc",
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

                odom = self.dataset.get_vehicle_odom_data(idx=start_idx)
                vels = self.ground_truth_encoder.encode(odom)

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
                odom = self.dataset.get_vehicle_odom_data(idx=i)
            else:
                odom = np.empty(0)
            
            try:
                camera_view = self.dataset.get_camera_frame(idx=i)
            except AssertionError:
                camera_view = np.empty(shape=(0))
            
            self.plotter.plot_compilation(
                input_data=adc_cube,
                input_encoder=self.input_encoder,
                runner=self.runner,
                decoder=self.decoder,
                gt_data=odom,
                ground_truth_encoder=self.ground_truth_encoder,
                camera_view=camera_view,
                axs=self.axs,
                show=False
            )

            #save the frame
            self.save_frame(clear_axs=True)
    