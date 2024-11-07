import numpy as np

class _GTEncoder:
    """Parent class for encoding model ground truth (output) data
    """

    def __init__(self) -> None:
        
        #flag to note whether a full encoding is ready or not
        #(for encoders that encode a series of frames)
        self.full_encoding_ready = False

        self.configure()

        return
    
    def configure(self):
        """Configure the ground truth encoder. Remaining functionality
        must be implemented by child class to configure its modules.
        """

        return

    def encode(self,gt_data:np.ndarray)->np.ndarray:
        """Implemented by child class to encode data for a specific
        model

        Args:
            gt_data (np.ndarray): output data expressed as a numpy array 
                (by default)

        Returns:
            np.ndarray: np.ndarray consisting of the output data
                encoding
        """
        pass

    def reset(self):
        """Implemented by child class to reset encoder history (if applicable)
        """
        self.full_encoding_ready = False
        return