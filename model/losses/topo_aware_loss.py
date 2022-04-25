import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch import Tensor, einsum

class VGG19_3Layer(nn.Module):

    def __init__(self):
        super(VGG19_3Layer, self).__init__()
        self.x64 = nn.Conv2d(1, 64, kernel_size=1, padding=0, bias=False)
        self.x128 = nn.Conv2d(1, 128, kernel_size=1, padding=0, bias=False)
        self.x256 = nn.Conv2d(1, 256, kernel_size=1, padding=0, bias=False)
        self.x64.weight = torch.nn.Parameter(torch.ones((64, 1, 1, 1)))
        self.x128.weight = torch.nn.Parameter(torch.ones((128, 1, 1, 1)))
        self.x256.weight = torch.nn.Parameter(torch.ones((256, 1, 1, 1)))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        cwd = os.getcwd()
        params = torch.load(os.path.join(cwd, 'model', 'losses', 'vgg19_bn', 'vgg19_bn-c79401a0.pth'))
        self.conv1_2.weight = torch.nn.Parameter(params['features.3.weight'])
        self.conv1_2.bias = torch.nn.Parameter(params['features.3.bias'])
        self.conv2_2.weight = torch.nn.Parameter(params['features.10.weight'])
        self.conv2_2.bias = torch.nn.Parameter(params['features.10.bias'])
        self.conv3_4.weight = torch.nn.Parameter(params['features.17.weight'])
        self.conv3_4.bias = torch.nn.Parameter(params['features.17.bias'])
        # {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for p in self.parameters():  # 冻结VGG的参数，参数更新时不更新TWLoss
            p.requires_grad = False

    def forward(self, x):
        x64 = self.x64(x)
        x64 = self.conv1_2(x64)
        x128 = self.x128(x)
        x128 = self.conv2_2(x128)
        x256 = self.x256(x)
        x256 = self.conv3_4(x256)
        x_vgg = torch.cat([x64, x128, x256], dim=1)
        return x_vgg
    
class TopoAwareLoss(nn.Module):
    def __init__(self):
        super(TopoAwareLoss, self).__init__()
        # VGG model for topology aware loss
        self._vgg19_3_model = VGG19_3Layer().cuda().eval()
        self._criterion_mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        pred_vgg19_3   = self._vgg19_3_model(torch.softmax(pred, dim=1)[:, 1, :, :].unsqueeze(1))
        target_vgg19_3 = self._vgg19_3_model(target.type(torch.cuda.FloatTensor))
        return self._criterion_mse(pred_vgg19_3, target_vgg19_3)