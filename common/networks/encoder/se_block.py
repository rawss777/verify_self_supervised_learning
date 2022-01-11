from torch import nn


class SEBlock(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SEBlock, self).__init__()
        hidden_planes = planes // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(planes, hidden_planes, kernel_size=1, stride=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_planes, planes, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.act(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x
