from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch
import numpy as np

import mmwave_model_integrator.torch_training.transforms as transforms
from mmwave_model_integrator.torch_training.datasets._base_torch_dataset import _BaseTorchDataset

class DopAzToVelDataset(_BaseTorchDataset):
    """Dataset for radar Doppler-Azimuth images to velocity vector prediction."""

    def __init__(self,
                 input_paths:list,
                 output_paths:list,
                 input_transforms:list = None,
                 output_transforms:list = None):
        """initialize the segmentation dataset

        Args:
            input_paths (list): list of paths (strings) to each input file
            output_paths (list): list of paths (strings) to each output file
            input_transforms (list, optional): A list of transforms to be applied to each input item when __getitem__ is called. Will be fed into a compose() method Defaults to None.
            output_transforms (list, optional):  A list of transforms to be applied to each output item when __getitem__ is called. Will be fed into a compose() method Defaults to None.
        """

        super().__init__(
            input_paths=input_paths,
            output_paths=output_paths,
            input_transforms=input_transforms,
            output_transforms=output_transforms
        )

        return
    
    def __getitem__(self, idx):
        # Load input radar image
        image_path = self.input_paths[idx]
        image = np.load(image_path).astype(np.float32)  # shape: [H, W] or [C, H, W]

        # Convert to torch.Tensor
        if image.ndim == 2:  # [H, W]
            image = torch.from_numpy(image).unsqueeze(0)  # -> [1, H, W]
        elif image.ndim == 3:
            image = torch.from_numpy(image)  # Already [C, H, W]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Apply additional transforms (e.g., normalization)
        if self.input_transforms:
            transform = Compose(self.input_transforms)
            image = transform(image)

        # Load target vector
        target_path = self.output_paths[idx]
        target = np.load(target_path).astype(np.float32)
        target = torch.from_numpy(target).float()  # shape [2]

        return image, target