import torch
from torch import nn

class QuantileRegressionLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(
                torch.max(
                    (q-1) * errors,
                    q * errors
                ).unsqueeze(2))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=2), dim=2))
        return loss
