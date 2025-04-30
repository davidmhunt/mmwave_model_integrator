from torch.utils.data import Dataset
from torchvision.transforms import Compose
import mmwave_model_integrator.torch_training.transforms as transforms
import torch
import numpy as np

class _BaseTorchDataset(Dataset):
    """The basic torch dataset used by mmWave model integrator

    Args:
        Dataset (_type_): _description_
    """
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
        self.input_paths = input_paths
        self.output_paths = output_paths

        #configure transforms
        self.input_transforms = []
        for transform_config in input_transforms:
            transform_class = getattr(transforms,transform_config['type'])
            transform_config.pop('type')
            self.input_transforms.append(
                transform_class(**transform_config)
            )
        
        self.output_transforms = []
        for transform_config in output_transforms:
            transform_class = getattr(transforms,transform_config['type'])
            transform_config.pop('type')
            self.output_transforms.append(
                transform_class(**transform_config)
            )

        self.num_samples = len(input_paths)

    def __len__(self):
        """Get the number of samples in the dataset

        Returns:
            int: number of samples in the dataset
        """
        
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a item from the dataset at the specified index

        Args:
            idx (int): _description_

        Returns:
            (torch.tensor,torch.tensor): a tuple containing the (image,mask)
        """

        #get the path to the sample image
        image_path = self.input_paths[idx]
        mask_path = self.output_paths[idx]

        #load the data from the disk
        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        #apply transforms as required
        if self.input_transforms:
            transforms = Compose(self.input_transforms)
            image = transforms(image)
        if self.output_transforms:
            transforms = Compose(self.output_transforms)
            mask = transforms(mask)

        return (image,mask)
