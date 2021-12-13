""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, conv1, conv2, conv3=None):
        super().__init__()
        self.maxpool_conv = [
            nn.MaxPool2d(2),
            conv1,
            nn.ReLU(inplace=True),
            conv2,
            nn.ReLU(inplace=True)
        ]
        if conv3:
            self.maxpool_conv += [
                conv3,
                nn.ReLU(inplace=True)
            ]
        self.maxpool_conv = nn.Sequential(*self.maxpool_conv)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = x2 + x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet16ResSKip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pretrained=True):
        super(UNet16ResSKip, self).__init__()

        self.encoder = models.vgg16(pretrained=pretrained).features

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Sequential(self.encoder[0],
                                 nn.ReLU(inplace=True),
                                 self.encoder[2],
                                 nn.ReLU(inplace=True)
                                 )
        self.down1 = Down(self.encoder[5], self.encoder[7])
        self.down2 = Down(self.encoder[10], self.encoder[12], self.encoder[14])
        self.down3 = Down(self.encoder[17], self.encoder[19], self.encoder[21])
        self.down4 = Down(self.encoder[24], self.encoder[26], self.encoder[28])
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
