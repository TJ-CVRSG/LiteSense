import torch
import torch.nn as nn

from .gradient import GradientLoss
from .scale import SILogLoss

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

        self.SILog = SILogLoss()
        self.Gradient = GradientLoss("Sobel")

    def forward(self, pred, dpt_gt):
        silog_loss = self.SILog(pred, dpt_gt, mask=((dpt_gt > 0)), interpolate=True)
        gradient_loss = self.Gradient(pred, dpt_gt)

        loss = 0.5 * silog_loss + 0.5 * gradient_loss
        loss_items = {
            "Loss-S": silog_loss,
            "Loss-G": gradient_loss,
        }

        return loss, loss_items
