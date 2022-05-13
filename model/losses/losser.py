import sys,os
sys.path.append(os.getcwd()+'/../../')

import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt

from typing import Callable, Iterable, List, Set, Tuple, Dict
from pathlib import Path

from .bcw_loss  import BalancedClassWeight
from .dice_loss import DiceLoss
from .uw_loss import UnetWeight
from utils import *


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
            label_one_hot = torch.stack([label == c for c in range(class_num)], dim=1).squeeze(2).type(torch.float32)
            loss = self.dice_criterion.forward(output.squeeze(2), label_one_hot)
        return loss


class WeightMapLoss(nn.Module):
    """
        Calculate weighted loss with weight maps
    """
    def forward(self, pred: Tensor, target: Tensor, weight_maps: Tensor, eps: float = 1e-20) -> Tensor:
        class_num = weight_maps.size()[1]
        mask = target.float()
        logit = torch.softmax(pred, dim=1)
        loss = 0
        weight_maps = weight_maps.float()
        for idx in range(class_num):
            if weight_maps.dim() == 4:
                loss += -1 * weight_maps[:, idx, :, :] * torch.log(logit[:, idx, :, :] + eps)
            elif weight_maps.dim() == 5:
                loss += -1 * weight_maps[:, idx, :, :, :] * torch.log(logit[:, idx, :, :, :] + eps)
        return loss.sum() / weight_maps.sum()


def prepare_weight_map(labels_path: Path, out_path: Path, func_names: List, class_num: int = 2) -> None:
    """
        Generate the weight map in data directory, such as balanced class weight or unet weight
    """
    file_paths = []
    for file in labels_path.rglob("*.npy"):
        if Path.is_file(file) and ".ipynb_checkpoints" not in str(str(file)):
            file_paths.append(file)
    file_paths.sort()
    # init
    bcw_weight = BalancedClassWeight(class_num)
    uw_weight = UnetWeight()
    print("Weight Map Calculation...")
    for file_path in file_paths:
        file_name = file_path.name
        print("Analyzing {} ..." .format(file_name))
        label = load_data_from_disk(file_path)
        for func_name in func_names:
            weight_path = Path(out_path, func_name)
            weight_path.mkdir(parents=True, exist_ok=True)
            weight = np.zeros((label.shape[0], label.shape[1], class_num))
            if func_name == 'bcw':
                weight = bcw_weight.get_weight(label)
            elif func_name == 'uw':
                assert class_num == 2, "The unet weight only support binary segmentation"
                weight = uw_weight.get_weight(label)
            print("{} for {} has done".format(func_name, file_name))
            print("Visualization:")
            for vis_idx in range(class_num):
                print("{} {}, the max value is {} and the min value is {}".format(func_name, vis_idx, 
                                                                                  np.amax(weight[:, :, vis_idx]),
                                                                                  np.amin(weight[:, :, vis_idx])))
                # print('    and the unique values are {}'.format(np.unique(weight[:, :, vis_idx])))
            plt.figure(figsize=(10, 10))
            plt.subplot(1, class_num + 1, 1), plt.imshow(label, cmap="gray"), plt.title("label"), plt.axis("off")
            for vis_idx in range(class_num):
                plt.subplot(1, class_num + 1, vis_idx + 2)
                plt.imshow(weight[:, :, vis_idx]), plt.title("{} {}".format(func_name, vis_idx)), plt.axis("off")
            plt.show()
            # print(str(Path(weight_path, file_name.replace('png', 'npy'))))
            np.save(str(Path(weight_path, file_name.replace("png", "npy"))), weight)
