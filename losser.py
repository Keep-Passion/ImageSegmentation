import os
import torch
import torch.nn as nn
from torch import Tensor
from model.losses.dice_loss import DiceLoss


class Losser:
    """
        Loss class for network training, you can add your loss in here
    """
    def __init__(self):
        self.wm_criterion   = WeightMapLoss()
        self.dice_criterion = DiceLoss()
    
    def get_loss(self, output: Tensor, label: Tensor, weight: Tensor, loss_name: str, class_num: int = 2) -> Tensor:
        if loss_name in ['bcw', 'uw']:
            loss = self.wm_criterion.forward(output, label, weight)
        if loss_name == 'dice':
            label_one_hot = torch.stack([label == c for c in range(class_num)], dim=1).squeeze(2).type(torch.int32).type(torch.float)
            loss = self.dice_criterion.forward(output.squeeze(2), label_one_hot)
        return loss


class WeightMapLoss(nn.Module):
    """
        Calculate weighted loss with weight maps
    """
    def forward(self, pred: Tensor, target: Tensor, weight_maps: Tensor, eps: float = 1e-20) -> Tensor:
        class_num = weight_maps.size()[2]
        mask = target.float()
        logit = torch.softmax(pred, dim=1)
        loss = 0
        weight_maps = weight_maps.float()
        for idx in range(class_num):
            if weight_maps.dim() == 4:
                loss += -1 * weight_maps[:, idx, :, :] * (torch.log(logit[:, idx, :, :]) + eps)
            elif weight_maps.dim() == 5:
                loss += -1 * weight_maps[:, idx, :, :, :] * (torch.log(logit[:, idx, :, :, :]) + eps)
        return loss.sum() / weight_maps.sum()
