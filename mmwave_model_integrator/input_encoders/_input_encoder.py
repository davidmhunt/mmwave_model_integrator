import numpy as np

class _InputEncoder:
    """Class for encoding model input data
    """

    def __init__(self) -> None:

        #flag to note whether a full encoding is ready or not
        #(for encoders that encode a series of frames)
        self.full_encoding_ready = False

        self.configure()
        
        return

    def configure(self):
        """Configure the encoder. Remaining functionality implemented by child class
        """
        pass

    def reset(self):
        """Implemented by child class to reset encoder history (if applicable)
        """
        self.full_encoding_ready = False
        return

    def encode(self,adc_data_cube:np.ndarray)->np.ndarray:
        """Implemented by child class to encode data for a specific model
        Args:
            adc_data_cube (np.ndarray): (rx antennas) x (adc samples) x
                (num_chirps) adc data cube consisting of complex data

        Returns:
            np.ndarray: np.ndarray consisting of data to be input
                into the model
        """
        
        pass