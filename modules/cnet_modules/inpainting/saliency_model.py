import torch
import torchvision
from torch import nn
from PIL import Image
import numpy as np
import os


# MICRO RESNET
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.resblock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        out = self.resblock(x)
        return out + x


class Upsample2d(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample2d, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode='nearest')
        return x


class MicroResNet(nn.Module):
    def __init__(self):
        super(MicroResNet, self).__init__()

        self.downsampler = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 8, kernel_size=9, stride=4),
            nn.InstanceNorm2d(8, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
        )

        self.residual = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=1, bias=False, groups=32),
            ResBlock(64),
        )

        self.segmentator = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 16, kernel_size=3),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            Upsample2d(scale_factor=2),
            nn.ReflectionPad2d(4),
            nn.Conv2d(16, 1, kernel_size=9),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.downsampler(x)
        out = self.residual(out)
        out = self.segmentator(out)
        return out
