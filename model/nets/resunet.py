# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


# *********************************** ResUNet *********************************************
class ResUNet(nn.Module):
    def __init__(self, num_channels=2, num_classes=2):
        super(ResUNet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]
        self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0]),
                                   Pooling(num_feat[0], num_feat[1]))
        self.down2 = nn.Sequential(Conv3x3_Res(num_feat[1], num_feat[1]),
                                   Pooling(num_feat[1], num_feat[2]))
        self.down3 = nn.Sequential(Conv3x3_Res(num_feat[2], num_feat[2]),
                                   Pooling(num_feat[2], num_feat[3]))
        self.down4 = nn.Sequential(Conv3x3_Res(num_feat[3], num_feat[3]),
                                   Pooling(num_feat[3], num_feat[4]))
      
        self.bottom = Conv3x3_Res(num_feat[4], num_feat[4])
        
        self.dconv1 = Conv3x3_Res(num_feat[4], num_feat[4])
        self.up1 = Unpooling(num_feat[4], num_feat[3])
        self.dconv2 = Conv3x3_Res(num_feat[3], num_feat[3])
        self.up2 = Unpooling(num_feat[3], num_feat[2])
        self.dconv3 = Conv3x3_Res(num_feat[2], num_feat[2])
        self.up3 = Unpooling(num_feat[2], num_feat[1])
        self.dconv4 = Conv3x3_Res(num_feat[1], num_feat[1])
        self.up4 = Unpooling(num_feat[1], num_feat[0])
        self.dconv5 = Conv3x3(num_feat[0], num_feat[0])
        self.final = nn.Conv2d(num_feat[0], num_classes, kernel_size=1)

    def forward(self, inputs, return_features=False):
        down1_feat= self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        down4_feat = self.down4(down3_feat)

        bottom_feat = self.bottom(down4_feat)
        
        dconv1_feat = self.dconv1(bottom_feat)
        up1_feat = self.up1(dconv1_feat,down4_feat)
        dconv2_feat = self.dconv2(up1_feat)
        up2_feat = self.up2(dconv2_feat,down3_feat)
        dconv3_feat = self.dconv3(up2_feat)
        up3_feat = self.up3(dconv3_feat,down2_feat)
        dconv4_feat = self.dconv4(up3_feat)
        up4_feat = self.up4(dconv4_feat,down1_feat)

        dconv5_feat = self.dconv5(up4_feat)
        outputs = self.final(dconv5_feat)
        return outputs

    
class Pooling(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(Pooling,self).__init__()        
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0),
                                  nn.BatchNorm2d(out_feat),
                                  nn.ReLU(inplace=True))

    def forward(self,inputs):
        outputs = self.bottleneck(inputs)
        outputs = self.pool(outputs)
        return outputs


class Unpooling(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Unpooling, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_feat,
                              out_feat,
                              kernel_size=2,
                              stride=2)

    def forward(self, inputs, mirrors): 
        inputs.add_(mirrors)
        outputs=self.deconv(inputs)
        return outputs


class Conv3x3_Res(nn.Module):
    def __init__(self, in_feat, out_feat,identify=False):
        super(Conv3x3_Res, self).__init__()
        self.identify=identify
        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())
        if self.identify:
            self.conv=nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1),
                                   nn.BatchNorm2d(out_feat))
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        if self.identify:
            inputs=self.conv(inputs)
        outputs.add_(inputs)
        return outputs

class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
