import numpy as np
import inspect
import torch
from torch.nn import Module

from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from mmwave_model_integrator.model_runner._model_runner import _ModelRunner

class GNNRunner(_ModelRunner):

    def __init__(
            self, 
            model, 
            state_dict_path, 
            cuda_device="cuda:0",
            use_sigmoid=True,
            edge_radius=5.0,
            enable_downsampling=False,
            downsample_keep_ratio=0.5,
            downsample_min_points=100
            ):
        
        self.edge_radius = edge_radius
        self.use_sigmoid = use_sigmoid
        
        #downsampling parameters
        self.enable_downsampling = enable_downsampling
        self.downsample_keep_ratio = downsample_keep_ratio
        self.downsample_min_points = downsample_min_points

        super().__init__(model, state_dict_path, cuda_device)
        
        #inspect the model to determine the input arguments
        self.forward_signature = inspect.signature(self.model.forward)

    def compute_edge_index(self,x:torch.Tensor)->torch.Tensor:
        """Compute the edge index for the given points

        Args:
            x (torch.Tensor): Nx4 Input tensor of points

        Returns:
            torch.Tensor: edge_index tensor
        """
        return radius_graph(x[:, :2], r=self.edge_radius, loop=False)

    def compute_edge_attr(self,x:torch.Tensor,edge_index:torch.Tensor)->torch.Tensor:
        """Compute the edge attributes for the given points

        Args:
            x (torch.Tensor): Nx4 Input tensor of points
            edge_index (torch.Tensor): edge_index tensor

        Returns:
            torch.Tensor: edge_attr tensor
        """
        row, col = edge_index
        src_pos = x[row, :2]
        dst_pos = x[col, :2]
        return (src_pos - dst_pos).norm(dim=1)

    def percentage_downsample(self,x, keep_ratio=0.5, min_points=100):
        """
        Randomly downsamples points.
        
        Args:
            x (Tensor): Node features [Num_Points, Features]
            keep_ratio (float): Fraction of points to keep (0.0 to 1.0)
            min_points (int): Minimum number of points to preserve.
            
        Returns:
            Tensor: downsampled_x
        """
        num_points = x.shape[0]
        
        # 1. Calculate Target Count
        target_count = int(num_points * keep_ratio)
        target_count = max(target_count, min_points)
        final_count = min(num_points, target_count)
        
        # 2. Random Sampling 
        if final_count < num_points:
            # Generate the random indices 
            choice = torch.randperm(num_points)[:final_count]
            
            # Apply the indices 
            x = x[choice]
        
        return x

    def make_prediction(self, nodes:np.ndarray):

        # Convert to PyTorch tensors
        x = torch.tensor(nodes, dtype=torch.float32)

        #downsample if enabled
        if self.enable_downsampling:
            x = self.percentage_downsample(
                x,
                keep_ratio=self.downsample_keep_ratio,
                min_points=self.downsample_min_points
            )
        
        #move to device
        x = x.to(self.device)

        #construct arguments for the forward pass
        model_args = {}
        
        #always add x
        model_args["x"] = x

        #add other arguments if they are in the signature
        if "edge_index" in self.forward_signature.parameters:
            model_args["edge_index"] = self.compute_edge_index(x).to(self.device)
        
        if "edge_attr" in self.forward_signature.parameters:
             #ensure edge index is computed
             if "edge_index" not in model_args:
                 model_args["edge_index"] = self.compute_edge_index(x).to(self.device)
             
             model_args["edge_attr"] = self.compute_edge_attr(x,model_args["edge_index"]).to(self.device)
        
        if "batch" in self.forward_signature.parameters:
            #create a batch of zeros
            model_args["batch"] = torch.zeros(x.shape[0],dtype=torch.int64).to(self.device)

        #perform the forard pass
        with torch.no_grad():
             pred = self.model(**model_args).squeeze()
        
        if self.use_sigmoid:
            pred = torch.sigmoid(pred)

        pred = pred.cpu().detach().numpy()
        
        #ensure x is back on cpu for indexing
        x = x.cpu().detach().numpy()

        return x[pred > 0.01,:]