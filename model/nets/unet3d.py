# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


# *********************************** UNet3D *********************************************
class UNet3D(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(UNet3D, self).__init__()
        init_channels = 64

        self.down1 = nn.Sequential(Conv3x3_3D(num_channels, init_channels))
                                   
        self.down2 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2)), 
                                   Conv3x3_3D(init_channels, 2 * init_channels))
                                   
        self.down3 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2)), 
                                   Conv3x3_3D(2 * init_channels, 4 * init_channels))
                                   
        self.bottom = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2)), 
                                    Conv3x3_3D(4 * init_channels, 8 * init_channels))

        self.upconcat1 = UpConcat_3D(4 * init_channels)
        self.upconv1   = Conv3x3_3D(4 * init_channels + 8 * init_channels, 4 * init_channels)

        self.upconcat2 = UpConcat_3D(2 * init_channels)
        self.upconv2   = Conv3x3_3D(2 * init_channels + 4 * init_channels, 2 * init_channels)

        self. upconcat3 = UpConcat_3D(init_channels)
        self.upconv3    = Conv3x3_3D(init_channels + 2 * init_channels, init_channels)

        self.final = nn.Sequential(nn.Conv3d(init_channels, num_classes, 1))

    def forward(self, inputs, return_features=False):
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        bottom_feat = self.bottom(down3_feat)

        up1_feat = self.upconcat1(bottom_feat, down3_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.upconcat2(up1_feat, down2_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.upconcat3(up2_feat, down1_feat)
        up3_feat = self.upconv3(up3_feat)

        if return_features:
            outputs = up3_feat
        else:
            outputs = self.final(up3_feat)
        return outputs


class Conv3x3_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3_3D, self).__init__()

        if in_channels < out_channels:
            # if in_channels < out_channels we're in the encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # otherwise we're in the decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.conv1 = nn.Sequential(nn.Conv3d(conv1_in_channels, conv1_out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.GroupNorm(32, conv1_out_channels),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv3d(conv2_in_channels, conv2_out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.GroupNorm(32, conv2_out_channels),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat_3D(nn.Module):
    def __init__(self, out_channels):
        super(UpConcat_3D, self).__init__()

        self.deconv = nn.ConvTranspose3d(2 * out_channels,
                           2 * out_channels,
                           kernel_size=3,
                           stride=(2, 2, 2),
                           padding=1,
                           output_padding=1)


    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out