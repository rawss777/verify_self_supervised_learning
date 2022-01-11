from typing import Union
import torch
from torch import nn


class ConvBnAct(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False, use_bn: bool = True, act: nn = nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_planes) if use_bn else nn.Identity()
        self.act = act(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SKBlock(nn.Module):
    def __init__(self, planes, reduction=16, path_num=2):
        super(SKBlock, self).__init__()
        hidden_planes = planes // reduction
        self.path_num = path_num

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(planes, hidden_planes, kernel_size=1, stride=1, bias=False)),
            ('bn', nn.BatchNorm2d(hidden_planes)),
            ('act', nn.ReLU(inplace=True))
        ]))
        self.conv2 = nn.Conv2d(hidden_planes, planes * self.path_num, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_paths):
        x = torch.sum(x_paths, dim=0)
        x = self.pool(x)
        x = self.conv1(x)

        attn = self.conv2(x)
        B, C, H, W = attn.shape
        attn = attn.view(B, self.path_num, C // self.path_num, H, W)  # shape = (b, self.path_num, c, h=1, w=1)
        attn = torch.permute(attn, (1, 0, 2, 3, 4))                   # shape = (self.path_num, b, c, h=1, w=1)
        attn = self.softmax(attn)
        x_paths *= attn
        return torch.sum(x_paths, dim=0)


class SKConv(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: Union[int, list], stride: int = 1,
                 dilation: int = 1, groups: int = 1, sk_reduction: int = 16):
        super(SKConv, self).__init__()
        kernel_size = kernel_size or [3, 5]  # default to one 3x3 and one 5x5 branch.
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2

        # 5x5 -> 3x3 + dilation
        dilation = [dilation * (k - 1) // 2 for k in kernel_size]
        kernel_size = [3] * len(kernel_size)
        self.path_list = nn.ModuleList([
            ConvBnAct(in_planes, out_planes, kernel_size=k, stride=stride, padding=d, dilation=d, groups=groups)
            for k, d in zip(kernel_size, dilation)
        ])

        # selective kernel
        self.sk = SKBlock(out_planes, sk_reduction, len(kernel_size))

    def forward(self, x):
        x_paths = [op(x) for op in self.path_list]
        x_paths = torch.stack(x_paths)  # shape = (len(self.path_list), b, c, h, w)
        x = self.sk(x_paths)
        return x
