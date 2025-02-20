import numpy as np

class _Decoder:
    """base decoder for decoding model outputs
    """
    def __init__(self)-> None:
    

        self.configure()

        return

    def configure(self):
        """Configure the decoder. 
        Remaining functionality must be implemented by 
        child class to configure its modules.
        """
        return

    def decode(self,model_prediction:np.ndarray)->np.ndarray:
        """Implemented by child class to decode data

        Args:
            model_prediction (np.ndarray): array consisting of 
                output data from a model

        Returns:
            np.ndarray: np.ndarray corresponding to the model output
        """
        return model_prediction