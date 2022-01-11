from collections import OrderedDict
from copy import deepcopy
import torch
from torch import nn
from .se_block import SEBlock


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, se: bool = False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, self.expansion * planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion * planes)
        self.act2 = nn.ReLU(inplace=True)
        self.se = SEBlock(self.expansion * planes) if se else nn.Identity()

        # short cut
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
                ("bn", nn.BatchNorm2d(self.expansion * planes))
            ]))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.act1(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, se: bool = False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.act3 = nn.ReLU(inplace=True)
        self.se = SEBlock(self.expansion * planes) if se else nn.Identity()

        # short cut
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
                ("bn", nn.BatchNorm2d(self.expansion * planes))
            ]))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, img_size: int, input_channel: int, block: str, num_blocks: list, wide_scale: int,
                 drop_rate: float, se: bool):
        super(ResNetEncoder, self).__init__()
        block = eval(block)
        self.se = se

        self.in_planes = 64 * wide_scale
        if img_size <= 32:  # is CIFAR
            self.entry_layer = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(input_channel, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn', nn.BatchNorm2d(self.in_planes)),
                ('act', nn.ReLU(inplace=True)),
            ]))
        else:
            self.entry_layer = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(input_channel, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(self.in_planes)),
                ('act1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(self.in_planes)),
                ('act2', nn.ReLU(inplace=True)),
            ]))

        self.layer1 = self._make_layer(block, 64 * wide_scale, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * wide_scale, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * wide_scale, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * wide_scale, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.output_dim = deepcopy(self.in_planes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append((f"block{idx+1}", block(self.in_planes, planes, stride, self.se)))
            self.in_planes = planes * block.expansion
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        out = self.entry_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.squeeze(-1).squeeze(-1)
        return out

