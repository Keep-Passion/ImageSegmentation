import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import matplotlib.pyplot as plt

class SignedDistanceMap():
    """
    the distance map referenced from:
        Xue Y, Tang H, Qiao Z, et al. Shape-Aware Organ Segmentation by Predicting Signed Distance Maps[J]. arXiv preprint arXiv:1912.03849, 2019.
    code copy from:
        https://github.com/JunMa11/SegWithDistMap/blob/b92df6f1937a2292b274741eb0e16006cf720049/code/train_LA_AAAISDF.py
    compute the signed distance map of binary mask
    input: label, shape = (h, w)
    output: the Signed Distance Map (SDM), shape = (h, w, 1)
    sdf(x) = 0; x in segmentation boundary
            -inf|x-y|; x in segmentation
            +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """    
    def _get_weight(self, label):
        normalized_sdf = np.zeros_like(label)
        posmask = label.astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            normalized_sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (np.max(posdis)-np.min(posdis))
            normalized_sdf[boundary == 1] = 0
        normalized_sdf = normalized_sdf[:, :, np.newaxis]
        return normalized_sdf


class SDMLoss(nn.Module):
    """
    the Loss referenced from:
        Xue Y, Tang H, Qiao Z, et al. Shape-Aware Organ Segmentation by Predicting Signed Distance Maps[J]. arXiv preprint arXiv:1912.03849, 2019.
    L = L_dice + u * (L_product + L_1)
    """
    def __init__(self, smooth=1e-5, u=10, k=1500):
        super(SDMLoss, self).__init__()
        self._smooth = smooth
        self._u = u
        self._k = k

    def forward(self, predict_sdm, target_sdm, target):
        # compute eq (4)
        predict_sdm = torch.tanh(predict_sdm)
        intersect = torch.sum(predict_sdm * target_sdm)
        pd_sum = torch.sum(predict_sdm ** 2)
        gt_sum = torch.sum(target_sdm ** 2)
        l_product = -1.0 * (intersect + self._smooth) / (intersect + pd_sum + gt_sum + self._smooth)
        l_sdf = l_product + torch.norm(predict_sdm - target_sdm, 1)/torch.numel(predict_sdm)
        output = torch.sigmoid(-1.0 * predict_sdm / self._k)
        loss = self._dice_loss(output, target) + self._u * l_sdf
        return loss
    
    def _dice_loss(self, predict, target):
        target = target.float()
        intersect = torch.sum(predict * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(predict * predict)
        loss = (2 * intersect + self._smooth) / (z_sum + y_sum + self._smooth)
        loss = 1 - loss
        return loss