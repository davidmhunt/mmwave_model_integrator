import numpy as np
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
            edge_radius=5.0):
        
        self.edge_radius = edge_radius

        super().__init__(model, state_dict_path, cuda_device)

    def make_prediction(self, nodes:np.ndarray):

        # Convert to PyTorch tensors
        x = torch.tensor(nodes, dtype=torch.float32)
        # Compute the edges using radius_graph (only x, y coordinates)
        edge_index = radius_graph(x[:, :2], r=self.edge_radius, loop=False)
        
        
        # x = torch.unsqueeze(x,0)
        x = x.to(self.device)
        # edge_index = torch.unsqueeze(edge_index,0)
        edge_index = edge_index.to(self.device)

        #perform the forard pass
        pred = self.model(x,edge_index).squeeze()
        pred = pred.cpu().detach().numpy()
        
        return nodes[pred > 0.50,:]