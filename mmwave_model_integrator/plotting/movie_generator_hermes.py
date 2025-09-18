import tqdm
import numpy as np

from cpsl_datasets.cpsl_ds import CpslDS


from mmwave_model_integrator.input_encoders.hermes_encoder import HermesEncoder
from mmwave_model_integrator.decoders._decoder import _Decoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders.hermes_gt_encoder import HermesGTEncoder
from mmwave_model_integrator.plotting.plotter_hermes import PlotterHermes

from mmwave_model_integrator.plotting._movie_generator import _MovieGenerator

class MovieGeneratorHermes(_MovieGenerator):

   
    def __init__(self,
                 cpsl_dataset:CpslDS,
                 plotter:PlotterHermes,
                 input_encoder:HermesEncoder,
                 runner:_ModelRunner=None,
                 decoder:_Decoder=None,
                 ground_truth_encoder:HermesGTEncoder=None,
                 temp_dir_path="~/Downloads/dop_az_to_pc",
                 ) -> None:
        
        self.dataset:CpslDS = cpsl_dataset
        self.input_encoder:HermesEncoder = input_encoder
        self.plotter:PlotterHermes = plotter

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

                try: #try accessing the full odometry data
                    vel_data = np.mean(self.dataset.get_vehicle_odom_data(start_idx)[:,8:11],axis=0)
                except AssertionError: #if not just get the x velocity (forward)
                    vel = np.mean(self.dataset.get_vehicle_vel_data(start_idx)[:,1])
                    vel_data = np.array([vel,0,0])
                    
                encoded_data = self.input_encoder.encode(
                    adc_data_cube=adc_cube,
                    vels=vel_data)

                if self.input_encoder.full_encoding_ready:
                    lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=start_idx)
                    pc = self.ground_truth_encoder.encode(lidar_pc,encoded_data)

                start_idx += 1
        else:
            while (not self.input_encoder.full_encoding_ready):
                
                try: #try accessing the full odometry data
                    vel_data = np.mean(self.dataset.get_vehicle_odom_data(start_idx)[:,8:11],axis=0)
                except AssertionError: #if not just get the x velocity (forward)
                    vel = np.mean(self.dataset.get_vehicle_vel_data(start_idx)[:,1])
                    vel_data = np.array([vel,0,0])

                #get the radar data
                adc_cube = self.dataset.get_radar_data(idx=start_idx)
                encoded_data = self.input_encoder.encode(
                    adc_data_cube=adc_cube,
                    vels=vel_data)

                start_idx += 1
        
        for i in tqdm.tqdm(range(start_idx,self.dataset.num_frames)):

            #get the adc cube
            adc_cube = self.dataset.get_radar_data(idx=i)

            try: #try accessing the full odometry data
                vel_data = np.mean(self.dataset.get_vehicle_odom_data(idx=i)[:,8:11],axis=0)
            except AssertionError: #if not just get the x velocity (forward)
                vel = np.mean(self.dataset.get_vehicle_vel_data(idx=i)[:,1])
                vel_data = np.array([vel,0,0])
            
            if self.ground_truth_encoder:
                lidar_pc = self.dataset.get_lidar_point_cloud_raw(idx=i)
            else:
                lidar_pc = np.empty(0)
            
            try:
                camera_view = self.dataset.get_camera_frame(idx=i)
            except AssertionError:
                camera_view = np.empty(shape=(0))
            
            
            self.plotter.plot_compilation(
                input_adc_cube=adc_cube,
                input_vels=vel_data,
                input_encoder=self.input_encoder,
                runner=self.runner,
                decoder=self.decoder,
                gt_data=lidar_pc,
                ground_truth_encoder=self.ground_truth_encoder,
                camera_view=camera_view,
                axs=self.axs,
                show=False
            )

            #save the frame
            if self.input_encoder.full_encoding_ready:
                self.save_frame(clear_axs=True)
    