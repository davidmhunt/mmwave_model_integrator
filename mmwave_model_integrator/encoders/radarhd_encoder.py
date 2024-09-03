import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.processors.range_azmith_resp import RangeAzimuthProcessor
from mmwave_model_integrator.encoders._radar_range_az_encoder import _RadarRangeAzEncoder

class RadarHDEncoder(_RadarRangeAzEncoder):

    def __init__(self,
                 config_manager: ConfigManager,
                 range_max:float = 10.8,
                 num_range_bins:int=256,
                 x_max:float=10,
                 y_max:float=10,
                 z_range:list=[-0.3,0.3],
                 mag_threshold:float = 0.05,
                 num_az_angle_bins:int = 64,
                 num_frames_history:int = 40
                 ) -> None:


        self.range_max:float = range_max
        self.num_range_bins:int = num_range_bins
        self.x_max:float = x_max
        self.y_max:float = y_max
        self.z_range:list = z_range
        self.mag_threshold:float = mag_threshold
        self.num_az_angle_bins:int = num_az_angle_bins

        #encoding the data with a specific amount of history stored
        self.num_frames_history=num_frames_history
        self.num_frames_encoded:int = None

        #array for latest encoded data (from parent class)
        #indexed by frame, rng bin, az bin
        self.encoded_data:np.ndarray = None 
        

        #seperate x and y mesh grid for dealing with the whole range az resp
        self.rhos_full:np.ndarray = None
        self.thetas_full:np.ndarray = None
        self.x_s_full:np.ndarray = None
        self.y_s_full:np.ndarray = None


        super().__init__(config_manager)

    def configure(self):
        
        #initialize the virtual array processor and range az response
        super().configure()

        #encoder is not ready on first encoding
        self.full_encoding_ready = False

        #determine the finalized set of range bins
        self.range_bins = \
            self.range_azimuth_processor.range_bins[:self.num_range_bins]

        #determine the angle bins to keep
        self.angle_bins = self.range_azimuth_processor.angle_bins
        
        #compute the mesh grid for plotting/encoding
        self.thetas,self.rhos = \
            np.meshgrid(self.angle_bins,
                        self.range_bins)
        self.x_s = np.multiply(self.rhos,np.cos(self.thetas))
        self.y_s = np.multiply(self.rhos,np.sin(self.thetas))

        #compute the mesh grid for dealing with the full range az resp
        self.thetas_full,self.rhos_full = \
            np.meshgrid(self.range_azimuth_processor.angle_bins,
                        self.range_azimuth_processor.range_bins)
        self.x_s_full = np.multiply(self.rhos_full,np.cos(self.thetas_full)).transpose()
        self.y_s_full =  np.multiply(self.rhos_full,np.sin(self.thetas_full)).transpose()

        #setup the final array to store the encoded data with frame
        #history
        self.encoded_data= np.zeros(
            shape=(self.num_frames_history+1,
                   self.num_range_bins,
                   self.num_az_angle_bins),
            dtype=np.uint8
        )
        self.num_frames_encoded = 0

    def encode(self,adc_data_cube:np.ndarray) -> np.ndarray:
        """Encoded a new frame of adc data for radarHD's model
        and add it to the full encoded data

        Args:
            adc_data_cube (np.ndarray): (rx antennas) x (adc samples) x
                (num_chirps) adc data cube consisting of complex data

        Returns:
            np.ndarray: (num_frames_history + 1) x (range bins) x 
            (az bins) array of an encoded frames for radarHD model
        """
        #increment the full encoded frame with history
        self.encoded_data[0:-1,:,:] = self.encoded_data[1:,:,:]

        #encode the latest data
        self.encoded_data[-1,:,:] = self.encode_new_frame(
            adc_data_cube
        )

        #increment the total count of encoded data
        self.num_frames_encoded += 1

        if self.num_frames_encoded >= self.num_frames_history + 1:
            self.full_encoding_ready = True

        return self.encoded_data

    
    def encode_new_frame(self, adc_data_cube: np.ndarray) -> np.ndarray:
        """Encoded a new frame of adc data for radarHD's model

        Args:
            adc_data_cube (np.ndarray): (rx antennas) x (adc samples) x
                (num_chirps) adc data cube consisting of complex data

        Returns:
            np.ndarray: (range bins) x (az bins) array of an encoded frame
        """
        adc_data_cube = self.virtual_array_reformater.process(
            adc_cube=adc_data_cube
        )

        #radarHD store range az resp as (az bins x rng bins)
        rng_az_resp = self.range_azimuth_processor.process(
            adc_cube=adc_data_cube,
            chirp_idx=0
        ).transpose()

        #threshold and convert to a point cloud
        pc = self.threshold(rng_az_resp)
        
        #convert to polar
        pc_polar = self.pcl_to_polar(pc)

        #generate the final encoded image
        frame_image_polar = self.create_image_polar(pc_polar)

        return frame_image_polar
    
    def reset_history(self):
        """Reset the encoded data and the tracking of the number of encoded frames
        """
        self.encoded_data= np.zeros(
            shape=(self.num_frames_history+1,
                   self.num_range_bins,
                   self.num_az_angle_bins),
            dtype=np.uint8
        )
        self.num_frames_encoded = 0

        return
    
    def get_rng_az_resp_from_encoding(self, rng_az_resp: np.ndarray) -> np.ndarray:
        """Given an encoded range azimuth response, return a single
        range azimuth response that can then be plotted. Implemented
        by child class

        Args:
            rng_az_resp (np.ndarray): encoded range azimuth response
                (frames) x (rng bins) x (az bins)

        Returns:
            np.ndarray: (range bins) x (az bins) range azimuth response
        """
        return rng_az_resp[-1,:,:]
    
    def threshold(self,rng_az_resp:np.ndarray)->np.ndarray:
        """RadarHD's threshold() function 

        Args:
            rng_az_resp (np.ndarray): (az bins) x (rng bins) range azimuth response

        Returns:
            np.ndarray: Nx4 point cloud of (x,y,z,intensity) values from a 
                thresholded set of points
        """

        m = np.max(rng_az_resp[:,6:])
        idx = (rng_az_resp[:,6:] >= self.mag_threshold*m)
        idx = np.concatenate(
            (np.zeros((self.num_az_angle_bins,6),
                      dtype=bool),idx),
                      axis=1)

        x = self.x_s_full[idx].reshape(-1,1)
        y = self.y_s_full[idx].reshape(-1,1)
        z = np.zeros(x.shape)
        intensity = rng_az_resp[idx].reshape(-1,1)

        pc = np.concatenate((x,y,z,intensity),axis=1)

        return pc
    
    def pcl_to_polar(self,point_cloud_cart:np.ndarray)->np.ndarray:
        """Convert a 4D (x,y,z,intensity) cartesian point cloud to
        a 4D (r,az,el,intensity) polar point cloud.
        Based on RadarHD's pcl_to_polar function

        Args:
            point_cloud_cart (np.ndarray): (x,y,z,intensity) point cloud
        Returns:
            np.ndarray: (range,az,el,intensity) point cloud (in radians)
        """

        #threshold the data first

        valid_idxs = (point_cloud_cart[:,0] > -1 * self.x_max) & \
                    (point_cloud_cart[:,0] <= self.x_max) & \
                    (point_cloud_cart[:,1] >= -1 * self.y_max) & \
                    (point_cloud_cart[:,1] <= self.y_max) & \
                    (point_cloud_cart[:,2] >= self.z_range[0]) & \
                    (point_cloud_cart[:,2] <= self.z_range[1])
        
        point_cloud_cart = point_cloud_cart[valid_idxs,:]

        #convert from cartesian to polar coordinates
        ranges = np.sqrt(point_cloud_cart[:, 0]**2 + point_cloud_cart[:, 1]**2 + point_cloud_cart[:, 2]**2)
        azimuths = np.arctan2(point_cloud_cart[:, 1], point_cloud_cart[:, 0])
        elevations = np.arccos(point_cloud_cart[:, 2] / ranges)
        intensities = point_cloud_cart[:,3]

        point_cloud_polar = np.column_stack(
            (ranges,azimuths,elevations,intensities))
        
        return point_cloud_polar
    

    def create_image_polar(self,point_cloud_polar):
        
        #threshold data to specified range and az limits
        valid_idxs = (point_cloud_polar[:,0] <= self.range_max) & \
                    (point_cloud_polar[:,1] >= np.deg2rad(-70)) & \
                    (point_cloud_polar[:,1] <= np.deg2rad(70))
        
        point_cloud_polar = point_cloud_polar[valid_idxs,:]

        #define the out grid
        out_grid = np.zeros((self.num_range_bins,self.num_az_angle_bins))

        #identify the nearest point from the pointcloud
        r_idx = np.argmin(np.abs(self.range_bins - point_cloud_polar[:,0][:,None]),axis=1)
        az_idx = np.argmin(np.abs(self.angle_bins - point_cloud_polar[:,1][:,None]),axis=1)

        #compute the intensity values in dB
        intensity = 10 * np.log10(point_cloud_polar[:,3])

        #normalize the intensity values
        min_intensity = np.min(intensity)
        max_intensity = np.max(intensity)
        intensity = (intensity - min_intensity) / \
                    (max_intensity - min_intensity)

        out_grid[r_idx,az_idx] = intensity

        #convert hte out_grid to be a np.uint8 value
        out_grid = (out_grid*255).astype(np.uint8)

        return out_grid