import numpy as np
from mmwave_model_integrator.ground_truth_encoders._gt_encoder_odom_to_vel import _GTEncoderOdomToVel

class RadVelGTEncoder(_GTEncoderOdomToVel):

    def __init__(self):
        super().__init__()

    def encode(self, gt_data):
        """Extended as needed by child class to encode gt data for a specific velocity encodings

        Args:
            gt_data (np.ndarray): Nx14 array containing odometry information with each row containing
                [time,x,y,z,quat_w,quat_x,quat_y,quat_z,vx,vy,vz,wx,wy,wz]

        Returns:
            np.ndarray: 2 element array with average [vx,vy] from a dataset
        """

        self.full_encoding_ready = True

        return np.mean(gt_data[:,8:10],axis=0)