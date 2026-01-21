import torch
import torch.nn as nn
import torch.nn.functional as F # <--- CORRECTED IMPORT

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, 1] - Raw logits from your model
        # targets: [N, 1] - Float labels (0.0 or 1.0)
        
        # 1. Compute standard BCE (per-pixel)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 2. Get probabilities (pt)
        pt = torch.exp(-bce_loss)
        
        # 3. Compute Focal Loss
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss