# https://github.com/zju3dv/deltar
import torch
import torch.nn as nn

class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=False):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
            
        g = torch.log(input + 1e-6) - torch.log(target + 1e-6)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        
        return 10 * torch.sqrt(Dg)
