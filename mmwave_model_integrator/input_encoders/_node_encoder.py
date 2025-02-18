import numpy as np
from mmwave_model_integrator.input_encoders._input_encoder import _InputEncoder

class _NodeEncoder(_InputEncoder):
    """Base Encoder for Node-based models (ex: GNNs)

    Args:
        _InputEncoder (_type_): _description_
    """
    def __init__(self):

        super().__init__()

        return
    
    def encode(self, nodes):
        """Implemented by child class to encode data for a specific node-based model

        Args:
            nodes (np.ndarray): NxM array of N nodes with M feature vectors

        Returns:
            np.ndarray: NxL array of N nodes with the finalized set of L feature vectors
        """
        return nodes

    