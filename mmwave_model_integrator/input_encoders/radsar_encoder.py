import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.synthetic_array_beamformer_processor_revA import SyntheticArrayBeamformerProcessor
from mmwave_radar_processing.detectors.CFAR import CaCFAR_1D,CaCFAR_2D

from mmwave_model_integrator.input_encoders._radar_range_az_encoder import _RadarRangeAzEncoder


class RadSAREncoder(_RadarRangeAzEncoder):

    def __init__(
            self,
            config_manager: ConfigManager,
            az_angle_bins_rad=\
                np.deg2rad(np.linspace(
                    start=-90,stop=90,num=90
                 )),
            min_vel=0.2,
            max_vel_change=0.1,
            mode:int = SyntheticArrayBeamformerProcessor.ENDFIRE_MODE,
            min_power_threshold_dB=40) -> None:
        
        
        #initialize a CFAR detector
        cfar_2d = CaCFAR_2D(
            num_guard_cells=np.array([2,5]),
            num_training_cells=np.array([3,15]),
            false_alarm_rate=0.001,
            resp_border_cells=np.array([5,5]),
            mode="full"
        )

        self.synthetic_array_processor = SyntheticArrayBeamformerProcessor(
            config_manager=config_manager,
            cfar=cfar_2d,
            az_angle_bins_rad=az_angle_bins_rad,
            el_angle_bins_rad=np.array([0]),
            min_vel=min_vel,
            max_vel_change=max_vel_change,
            mode=mode
        )

        self.num_az_angle_bins = az_angle_bins_rad.shape[0]

        #thresholding out low power parts of the response
        self.min_power_threshold_dB = min_power_threshold_dB

        #array for latest encoded data (from parent class)
        #NOTE: indexed by range bin, az bin, chirp idx
        self.encoded_data:np.ndarray = None

        super().__init__(config_manager)

        return
    
    def configure(self):

        #configure virtual array processors and range az response
        super().configure()

        #determine the finalized set of range bins
        self.range_bins = self.synthetic_array_processor.range_bins

        #determine the angle bins to keep
        self.angle_bins = self.synthetic_array_processor.az_angle_bins_rad

        #compute the mesh grid
        self.x_s = self.synthetic_array_processor.x_s[:,:,0]
        self.y_s = self.synthetic_array_processor.y_s[:,:,0]
        self.thetas = self.synthetic_array_processor.thetas[:,:,0]
        self.rhos = self.synthetic_array_processor.rhos[:,:,0]
        
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

        #generate the array geometry
        self.synthetic_array_processor.generate_array_geometries(vels)

        if self.synthetic_array_processor.array_geometry_valid:

            #compute the response
            resp = self.synthetic_array_processor.process(adc_data_cube)

            #get the azimuth response slice
            resp = resp[:,:,0]

            #convert to dB
            resp = 20 * np.log10(np.abs(resp))

            #perform thresholding
            thresholded_val = np.max(resp) - self.min_power_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val

            self.encoded_data = resp[:,:,np.newaxis]
            self.full_encoding_ready = True
            return self.encoded_data
        else:
            self.encoded_data = None
            self.full_encoding_ready = False
            return None

    
    def reset(self):
        """No history, method isn't used for this encoder
        """

        #reset the synthetic array processor
        self.synthetic_array_processor.last_vel = np.array([0,0,0])
        self.synthetic_array_processor.array_geometry_valid = False

        return super().reset()
    
    def get_rng_az_resp_from_encoding(self, rng_az_resp: np.ndarray) -> np.ndarray:
        """Given an encoded range azimuth response, return a single
        range azimuth response that can then be plotted. Implemented
        by child class

        Args:
            rng_az_resp (np.ndarray): encoded range azimuth response
                (rng bins) x (az bins)

        Returns:
            np.ndarray: (range bins) x (az bins) range azimuth response
        """
        return rng_az_resp[:,:,0]