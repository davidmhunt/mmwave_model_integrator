import numpy as np
from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder


class _GTNodeEncoder(_GTEncoder):
    """Parent class for encoding node-based models ground truth (output) data
    """

    def __init__(self):
        super().__init__()

        #set the encoding ready to be true
        self.full_encoding_ready = True

        return
    
    def encode(self, gt_data:np.ndarray)->np.ndarray:
        """Implemented by child class to encode gt data for a specific node-based model

        Args:
            gt_node_labels (np.ndarray): N-element array of N nodes with M feature vectors

        Returns:
            np.ndarray: N-element array of labels for each node
        """

        #set the full encoding ready to true
        self.full_encoding_ready = True

        return gt_data