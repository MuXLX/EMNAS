import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F

class Mix_Kernel(nn.Module):
    def __init__(self, in_channels, out_channels,stride, bias=0, affine=True):
        super(Mix_Kernel, self).__init__()
        self.stride = stride
        # self.bias = bias
        self.groups = in_channels
        self.r1 = nn.ReLU(inplace=False)
        self.weight1_1 = nn.Parameter(torch.randn(in_channels, in_channels // in_channels, 5, 5).cuda(), requires_grad=True)
        self.weight1_2 = nn.Parameter(torch.randn(in_channels, in_channels, 1, 1).cuda(), requires_grad=True)
        self.b1 = nn.BatchNorm2d(in_channels, affine=affine)

        self.r2 = nn.ReLU(inplace=False)
        self.weight2_1 = nn.Parameter(torch.randn(in_channels, in_channels // in_channels, 5, 5).cuda(), requires_grad=True)
        self.weight2_2 = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1).cuda(), requires_grad=True)
        self.b2 = nn.BatchNorm2d(in_channels, affine=affine)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, o):
        if o == 0:
            x = self.r1(x)
            x = F.conv2d(x, self.weight1_1[:,:,1:4,1:4],stride=self.stride,padding=1, groups=self.groups,bias=self.bias)
            x = F.conv2d(x, self.weight1_2, stride=1, padding=0, bias=self.bias)
            x = self.b1(x)
            x = self.r2(x)
            x = F.conv2d(x, self.weight2_1[:, :, 1:4, 1:4], stride=1, padding=1, groups=self.groups, bias=self.bias)
            x = F.conv2d(x, self.weight2_2, stride=1, padding=0, bias=self.bias)
            x = self.b2(x)
        elif o == 1:
            x = self.r1(x)
            x = F.conv2d(x, self.weight1_1, stride=self.stride, padding=2, groups=self.groups, bias=self.bias)
            x = F.conv2d(x, self.weight1_2, stride=1, padding=0, bias=self.bias)
            x = self.b1(x)
            x = self.r2(x)
            x = F.conv2d(x, self.weight2_1, stride=1, padding=2, groups=self.groups, bias=self.bias)
            x = F.conv2d(x, self.weight2_2, stride=1, padding=0, bias=self.bias)
            x = self.b2(x)
        elif o == 2:
            # x = self.r1(x)
            # x = F.conv2d(x, self.weight1_1[:, :, 1:4, 1:4], stride=self.stride, padding=2, dilation=2,
            #              groups=self.groups, bias=self.bias)
            # x = F.conv2d(x, self.weight1_2, stride=1, padding=0, bias=self.bias)
            # x = self.b1(x)

            x = self.r2(x)
            x = F.conv2d(x, self.weight2_1[:, :, 1:4, 1:4], stride=self.stride, padding=2, dilation=2,
                         groups=self.groups, bias=self.bias)
            x = F.conv2d(x, self.weight2_2, stride=1, padding=0, bias=self.bias)
            x = self.b2(x)

        elif o == 3:
            # x = self.r1(x)
            # x = F.conv2d(x, self.weight1_1, stride=self.stride, padding=4, dilation=2, groups=self.groups,
            #              bias=self.bias)
            # x = F.conv2d(x, self.weight1_2, stride=1, padding=0, bias=self.bias)
            # x = self.b1(x)

            x = self.r2(x)
            x = F.conv2d(x, self.weight2_1, stride=self.stride, padding=4, dilation=2, groups=self.groups,
                         bias=self.bias)
            x = F.conv2d(x, self.weight2_2, stride=1, padding=0, bias=self.bias)
            x = self.b2(x)

        return x

