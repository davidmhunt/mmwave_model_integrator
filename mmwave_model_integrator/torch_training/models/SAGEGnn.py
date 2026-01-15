import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SageGNNClassifier(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=16, out_channels=1,**kwargs):
        super(SageGNNClassifier, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)  # Output probability between 0 and 1