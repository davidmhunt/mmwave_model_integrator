import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import Compose
import mmwave_model_integrator.torch_training.transforms as ds_transforms
import numpy as np

from geometries.transforms.transformation import Transformation
from geometries.pose.orientation import Orientation
from geometries.coordinate_systems.coordinate_system_conversions import cartesian_to_cylindrical

class _GnnNodeDataset(Dataset):
    def __init__(self,
                 node_paths: list,
                 label_paths: list,
                 edge_radius: float = 5.0,
                 enable_random_yaw_rotate=False,
                 enable_occupancy_grid_preturbations=False,
                 enable_x_y_position_preturbations=False,
                 enable_cylindrical_encoding=False,
                 transforms: list = None):
        """Initialize the segmentation dataset with mandatory list-based transforms.

        Args:
            node_paths (list): List of paths to each node file
            label_paths (list): List of paths to each label file
            edge_radius (float, optional): The radius to use when clustering the nodes
            transforms (list,optional): List of dictionaries used to configure transforms. Defaults to None
            enable_random_yaw_rotate (bool,optional): On true, applies random yaw
                roations to data. Defaults to False
            enable_occupancy_grid_preturbations (bool,optional): On true, 
                applies random preturbations to the occupancy grid. Defaults
                to False
            enable_x_y_position_preturbations (bool,optional): On true, 
                applies random preturbations to the x,y position of each node
                in the grid. Defaults to False,
            enable_cylindrical_encoding (bool,optional): On true,
                graph features are additionally encoded in cylindrical coordinates (r,theta,z)
                instead of cartesian coordinates (x,y,z)
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
        else:
            self.transforms = None
        
        #statuses for data augmentations
        self.enable_random_yaw_rotate=enable_random_yaw_rotate
        self.enable_occupancy_grid_preturbations=enable_occupancy_grid_preturbations
        self.enable_x_y_position_preturbations=enable_x_y_position_preturbations
        self.enable_cylindrical_encoding = enable_cylindrical_encoding
    ####################################################################
    #Torch Dataset core functions
    #################################################################### 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load the data from disk
        nodes_path = self.node_paths[idx]
        labels_path = self.label_paths[idx]
        
        nodes = np.load(nodes_path).astype(np.float32)  # Nx4 array [x, y, z, probability]
        labels = np.load(labels_path).astype(np.float32)  # N-element array of labels
        
        #apply augmentations
        nodes = self.apply_augmentations(nodes)

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
    
    ####################################################################
    #Data augmentations
    ####################################################################

    def apply_augmentations(self,nodes:np.ndarray)->np.ndarray:
        """General function to apply augmentations based on which are applied

        Args:
            nodes (np.ndarray): Nx4 array of nodes with [x, y, z, probability]

        Returns:
            np.ndarray: Nodes array with preturbations and augmentations applied
        """

        if self.enable_random_yaw_rotate:
            nodes[:,0:3] = self.pc_random_yaw_rotate(nodes[:,0:3])
        if self.enable_occupancy_grid_preturbations:
            nodes[:,3] = self.preturb_occupancy_grid_probabilities(
                probabilities=nodes[:,3],
                sigma=0.05
            )
        if self.enable_x_y_position_preturbations:
            nodes[:,0:3] = self.preturb_pc_x_y_positions(nodes[:,0:3])

        if self.enable_cylindrical_encoding:
            cylindrical_encoding = self.pc_convert_to_cylindrical(
                pc_cart=nodes[:,0:3]
            )
            nodes = np.hstack((nodes,cylindrical_encoding[:,0:2]))
        
        return nodes


    def pc_random_yaw_rotate(self,pc:np.ndarray)->np.ndarray:
        """Apply a random yaw rotation to a given 3D point cloud

        Args:
            pc (np.ndarray): Nx3 Input point cloud of at least [x,y,z] points

        Raises:
            ValueError: If point cloud does not have correct dimmensions

        Returns:
            np.ndarray: Point cloud with a random rotation applied to it
        """
        if (pc.ndim == 2 and pc.shape[1] == 3):
            #come up with random rotation
            rng = np.random.default_rng()
            #declare a random rotation:
            rotation = Orientation.from_euler(
                yaw=(rng.random() * 359),
                degrees=True)
            transformation = Transformation(
                rotation=rotation._orientation
            )
            return transformation.apply_transformation(pc)


        else:
            raise ValueError("pc_random_yaw_rotate: input points must be an Nx3 array of points.")

    def preturb_occupancy_grid_probabilities(
            self,
            probabilities:np.ndarray,
            sigma:float=0.05)->np.ndarray:
        """Apply a random preturbation to the occcupancy grid probabilities

        Args:
            probabilities (np.ndarray): Nx1 or Nx1 element array of occupancy grid probabilities
            sigma (float, optional): standard deviation of preturbations to apply. Defaults to 0.05.

        Returns:
            np.ndarray: occupancy grid probabilities with preturbations applied
        """
        rng = np.random.default_rng()
        perturbations = rng.normal(
            loc=0,
            scale=sigma,
            size=probabilities.shape
            )
        
        return np.clip(probabilities + perturbations,a_min=0.05,a_max=1.0)


    def preturb_pc_x_y_positions(
            self,
            pc:np.ndarray,
            sigma:float=0.16)->np.ndarray:
        """Apply a random preturbation in x and y to given 3D point cloud

        Args:
            pc (np.ndarray): Nx3 Input point cloud of at least [x,y] points

        Raises:
            ValueError: If point cloud does not have correct dimmensions

        Returns:
            np.ndarray: Point cloud with a random preturbations in x,y applied to it
        """
        if (pc.ndim == 2 and pc.shape[1] >= 2):
            #come up with random preturbations
            rng = np.random.default_rng()
            perturbations = rng.normal(
                loc=0,
                scale=sigma,
                size=pc[:,0:2].shape
                )
            
            #apply the perturbations
            pc[:,0:2] += perturbations
            return pc


        else:
            raise ValueError("preturb_pc_x_y_positions: input points must be at least a Nx2 array of points.")
    
    def pc_convert_to_cylindrical(self,pc_cart:np.ndarray)->np.ndarray:
        """Converts a 3D point cloud in cartesian coordinates (x,y,z) to
            a cylindrical coordinates (r,theta,z)

        Args:
            pc_cart (np.ndarray): Nx3 Input point cloud of [x,y,z] points

        Raises:
            ValueError: If point cloud does not have correct dimmensions

        Returns:
            np.ndarray: Point cloud in cylindrical coordinates (r,theta,z)
        """
        if (pc_cart.ndim == 2 and pc_cart.shape[1] == 3):
            
            return cartesian_to_cylindrical(
                pc_cart
            )


        else:
            raise ValueError("pc_random_yaw_rotate: input points must be an Nx3 array of points.")