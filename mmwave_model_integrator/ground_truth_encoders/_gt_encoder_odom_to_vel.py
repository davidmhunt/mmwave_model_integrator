import numpy as np
from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder

class _GTEncoderOdomToVel(_GTEncoder):
    """
    Parent class for encoding velocity information from odometry data
    """

    def __init__(self):
        super().__init__()

        #set the encoding ready to be true
        self.full_encoding_ready = False

    def encode(self, gt_data:np.ndarray)->np.ndarray:
        """Extended as needed by child class to encode gt data for a specific velocity encodings

        Args:
            gt_data (np.ndarray): Nx14 array containing odometry information with each row containing
                [time,x,y,z,quat_w,quat_x,quat_y,quat_z,vx,vy,vz,wx,wy,wz]

        Returns:
            np.ndarray: Encoding of the specific velocity information required by a model
        """
        pass