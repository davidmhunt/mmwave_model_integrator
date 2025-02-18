import numpy as np
from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder


class _GTNodeEncoder(_GTEncoder):
    """Parent class for encoding node-based models ground truth (output) data
    """

    def __init__(self):
        super().__init__()

        return
    
    def encode(self, gt_data:np.ndarray)->np.ndarray:
        """Implemented by child class to encode gt data for a specific node-based model

        Args:
            gt_node_labels (np.ndarray): N-element array of N nodes with M feature vectors

        Returns:
            np.ndarray: N-element array of labels for each node
        """

        return gt_data