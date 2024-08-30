import numpy as np

class _lidarPCPolarDecoder:
    """base decoder for model outputs with lidar data encoded in a quantized
    polar grid format
    """
    def __init__(self)-> None:
    
        #range parameters
        self.range_bins:np.ndarray = None

        #angle parameters
        self.angle_bins:np.ndarray = None
        self.configure()

        return

    def configure(self):
        """Configure the lidar data decoder. 
        Remaining functionality must be implemented by 
        child class to configure its modules.
        """
        pass

    def decode(self,model_prediction:np.ndarray)->np.ndarray:
        """Implemented by child class to decode lidar data from
          a specific model

        Args:
            model_prediction (np.ndarray): array consisting of 
                output data from a model

        Returns:
            np.ndarray: np.ndarray Nx2 array of points from the decoded
                model output
        """
        pass

    def convert_polar_to_cartesian(self,points_polar:np.ndarray)->np.ndarray:

        """Convert an array of points stored as (range, azimuth) to (x,y)

        Args:
            points_polar (np.ndarray): Nx2 matrix of points in spherical (range,azimuth)

        Returns:
            (np.ndarray): Nx2 matrix of  points in cartesian (x,y)
        """

        x = points_polar[:,0] *  np.cos(points_polar[:,1])
        y = points_polar[:,0] *  np.sin(points_polar[:,1])


        return np.column_stack((x,y))