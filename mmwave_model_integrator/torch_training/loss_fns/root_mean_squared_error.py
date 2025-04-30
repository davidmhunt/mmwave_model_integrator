import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # To avoid sqrt(0)

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)
