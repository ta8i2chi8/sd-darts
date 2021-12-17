import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.modules.loss._Loss):
    def __init__(self, weight=0.1, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input, target, alpha):
        loss1 = F.cross_entropy(input, target)
        loss2 = -F.mse_loss(alpha, torch.tensor(0.5, requires_grad=False).cuda())
        return loss1 + self.weight * loss2, loss1.item(), loss2.item()
