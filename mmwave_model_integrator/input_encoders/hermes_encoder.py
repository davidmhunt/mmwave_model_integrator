import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.simple_synthetic_array_beamformer_processor_multiFrame import SyntheticArrayBeamformerProcessor

from mmwave_model_integrator.input_encoders._input_encoder import _InputEncoder


class HermesEncoder(_InputEncoder):

    def __init__(
            self,
            config_manager: ConfigManager,
            az_angle_bins_rad=np.deg2rad(np.linspace(
                start=-90,stop=90,num=90
            )),
            num_frames=5,
            stride=1,
            min_vel=np.array([0.00,0.30,0.0]),
            max_vel=np.array([0.05,0.70,0.05]), #np.array([0.05,0.50,0.05]),
            max_vel_stdev=np.array([0.02,0.02,0.02]), #np.array([0.02,0.02,0.02]),
            power_threshold_dB=40) -> None:
        
    

        self.synthetic_array_processor = SyntheticArrayBeamformerProcessor(
            config_manager=config_manager,
            az_angle_bins_rad=az_angle_bins_rad,
            el_angle_bins_rad=np.array([0]),
            chirp_cfg_idx=0,
            num_frames=num_frames, #10
            stride=stride,
            receiver_idx = 0,
            min_vel=min_vel,
            max_vel=max_vel,
            max_vel_stdev=max_vel_stdev,
            enable_calibration=False,
            num_calibration_iters=1
        )

        #thresholding for filtering out lower power signals
        self.power_threshold_dB = power_threshold_dB

        #array for latest encoded data (from parent class)
        self.encoded_data:np.ndarray = None

        #mesh grid for interpolated response
        self.interp_x_s = None
        self.interp_y_s = None

        super().__init__()

        return
    
    def configure(self):

        #configure virtual array processors and range az response
        super().configure()

        #compute the interpolated mesh grid
        self.interp_x_s = self.synthetic_array_processor.interp_x_s
        self.interp_y_s = self.synthetic_array_processor.interp_y_s
        
        return
    
    def encode(self, adc_data_cube: np.ndarray,vels:np.ndarray) -> np.ndarray:
        """Encode a synthetic array response

        Args:
            adc_data_cube (np.ndarray): (rx antennas) x (adc samples) x
                (num_chirps) adc data cube consisting of complex data
            vels (np.ndarray): 3-element [x,y,z] velocity vector

        Returns:
            np.ndarray: np.ndarray consisting of data to be input
                into the model
        """

        #process the response
        self.synthetic_array_processor.process(
            adc_cube=adc_data_cube,
            current_vel=vels
        )

        if self.synthetic_array_processor.array_geometry_valid:

            #get the interpolated response
            resp_interpolated = self.synthetic_array_processor.interpolated_beamformed_resp
            resp_interpolated = 20 * np.log10(np.abs(resp_interpolated) + 1e-12)

            #get the response (non-interpolated) to capture the noise floor
            resp = self.synthetic_array_processor.beamformed_resp
            resp = 20 * np.log10(np.abs(resp)) + 1e-12

            #low threshold - based on noise floor
            noise_dB = np.percentile(resp,q=30)
            low_threshold_val = noise_dB + self.power_threshold_dB
            idxs = resp_interpolated <= low_threshold_val
            resp_interpolated[idxs] = low_threshold_val

            #high threshold - to clip extremely high values
            high_threshold_val = low_threshold_val + 30
            idxs = resp_interpolated >= high_threshold_val
            resp_interpolated[idxs] = high_threshold_val


            #perform thresholding (old)
            # thresholded_val = np.max(resp) - self.min_power_threshold_dB

            

            #normalize to [0,1]
            resp_interpolated = (resp_interpolated - np.min(resp_interpolated)) / (np.max(resp_interpolated) - np.min(resp_interpolated))

            self.encoded_data = resp_interpolated
            self.full_encoding_ready = True
        else:
            self.encoded_data = None
            self.full_encoding_ready = False

        return self.encoded_data
    
    def reset(self):
        """No history, method isn't used for this encoder
        """

        #reset the synthetic array processor
        self.synthetic_array_processor.configure()

        return super().reset()