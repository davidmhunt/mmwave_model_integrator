import torch
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv

class EdgeConvGNNClassifier(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=16, out_channels=1):
        super(EdgeConvGNNClassifier, self).__init__()
        
        # EdgeConv layer expects a custom neural network to process edge features
        self.conv1 = EdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, hidden_channels),  # EdgeConv works with 2 concatenated node features
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        
        self.conv2 = EdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),  # Again using concatenated node features
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))

        self.conv3 = EdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, out_channels),
            torch.nn.Sigmoid()  # Output a probability between 0 and 1
        ))

    def forward(self, x, edge_index, edge_attr, **kwargs):
        # Perform edge convolution (message passing)
        x1 = self.conv1(x, edge_index, edge_attr)  # Use node features and edge features (edge_attr)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index, edge_attr)  # Final layer
        return x3  # Return output as probabilities (0-1)

