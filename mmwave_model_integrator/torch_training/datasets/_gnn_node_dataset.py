import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import Compose
import mmwave_model_integrator.torch_training.transforms as ds_transforms
import numpy as np

class _GnnNodeDataset(Dataset):
    def __init__(self,
                 node_paths: list,
                 label_paths: list,
                 edge_radius: float = 5.0,
                 transforms: list = None):
        """Initialize the segmentation dataset with mandatory list-based transforms.

        Args:
            node_paths (list): List of paths to each node file
            label_paths (list): List of paths to each label file
            edge_radius (float, optional): The radius to use when clustering the nodes
            transforms (list,optional): List of PyTorch Geometric transforms. Defaults to None
        """
        self.node_paths = node_paths
        self.label_paths = label_paths
        self.num_samples = len(node_paths)
        self.edge_radius = edge_radius

        # Ensure transforms is always a list; if empty, provide an empty list
        if transforms:
            self.transforms = []
            for transform_config in transforms:
                transform_class = getattr(ds_transforms,transform_config['type'])
                transform_config.pop('type')
                self.transforms.append(
                    transform_class(**transform_config)
                )

        self.transforms = transforms if transforms else []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load the data from disk
        nodes_path = self.node_paths[idx]
        labels_path = self.label_paths[idx]
        
        nodes = np.load(nodes_path).astype(np.float32)  # Nx4 array [x, y, z, probability]
        labels = np.load(labels_path).astype(np.float32)  # N-element array of labels

        # Convert to PyTorch tensors
        x = torch.tensor(nodes, dtype=torch.float32)

        # Compute the edges using radius_graph (only x, y coordinates)
        edge_index = radius_graph(x[:, :2], r=self.edge_radius, loop=False)

        # Compute edge attributes (Euclidean distance between nodes)
        edge_attr = []
        for i, j in edge_index.t():  # Iterate over edge pairs (i, j)
            dist = torch.norm(x[i, :2] - x[j, :2])  # Euclidean distance in 2D (x, y)
            edge_attr.append(dist)

        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # Convert to tensor

        # Convert labels to tensor
        y = torch.tensor(labels, dtype=torch.float32)

        # Create a PyTorch Geometric Data object with edge attributes
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Apply transforms dynamically for each sample
        if self.transforms:
            transforms = Compose(self.transforms)
            data = transforms(data)

        return data
