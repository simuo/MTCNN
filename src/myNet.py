import torch.nn as nn
import torch
import torchvision
from PIL import Image
import numpy as np
import numba


class P12Net(nn.Module):
    @numba.jit()
    def __init__(self):
        super(P12Net, self).__init__()
        self.layer = nn.Sequential(
            MobileNet(3, 10),
            DownSample(10, 2),
            MobileNet(10, 16),
            MobileNet(16, 32),
        )
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)  # conv4
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)  # conv5
        self.conv4_3 = nn.Conv2d(32, 1, 1, 1)

    @numba.jit()

    def forward(self, x):
        output = self.layer(x)
        conf, _ = torch.max(torch.sigmoid(self.conv4_1(output)), dim=1, keepdim=True)
        offset = self.conv4_2(output)
        iou = self.conv4_3(output)
        return conf, offset, iou


class R24Net(nn.Module):
    @numba.jit()
    def __init__(self):
        super(R24Net, self).__init__()
        self.layer = nn.Sequential(
            MobileNet(3, 28),
            DownSample(28, 2),
            MobileNet(28, 48),
            DownSample(48, 2),
            nn.Conv2d(48, 64, 2, 1),
            nn.PReLU(64)
        )
        self.linear_0 = nn.Linear(576, 128)
        self.prelu = nn.PReLU(128)

        self.linear_1 = nn.Linear(128, 2)
        self.linear_2 = nn.Linear(128, 4)
        self.linear_3 = nn.Linear(128, 1)

    @numba.jit()
    def forward(self, x):
        output = self.layer(x)
        output = output.view(output.size(0), -1)
        output = self.linear_0(output)
        output = self.prelu(output)

        conf = self.linear_1(output)
        offset = self.linear_2(output)
        iou = self.linear_3(output)
        conf, _ = torch.max(torch.sigmoid(conf), dim=1, keepdim=True)
        return conf, offset, iou


class O48Net(nn.Module):
    @numba.jit()
    def __init__(self):
        super(O48Net, self).__init__()
        self.layer = nn.Sequential(
            MobileNet(3, 32),
            DownSample(32, 2),
            MobileNet(32, 64),
            DownSample(64, 2),
            MobileNet(64, 64),
            DownSample(64, 2),
            nn.Conv2d(64, 128, 2, 1),
            MobileNet(128, 256)
        )
        self.linear_0 = nn.Linear(256, 128)
        self.prelu = nn.PReLU(128)

        self.linear_1 = nn.Linear(128, 2)
        self.linear_2 = nn.Linear(128, 4)
        self.linear_3 = nn.Linear(128, 1)

    @numba.jit()
    def forward(self, x):
        output = self.layer(x)
        output = output.view(output.size(0), -1)
        output = self.prelu(self.linear_0(output))

        conf = self.linear_1(output)
        conf, _ = torch.max(torch.sigmoid(conf), dim=1, keepdim=True)
        offset = self.linear_2(output)
        iou = self.linear_3(output)
        return conf, offset, iou

class MobileNet(nn.Module):
    @numba.jit()
    def __init__(self, in_channels, out_channels):
        super(MobileNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      groups=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=1),
            nn.PReLU(out_channels)
        )

    @numba.jit()
    def forward(self, x):
        return self.layer(x)

class DownSample(nn.Module):
    @numba.jit()
    def __init__(self, channels, kernel_size):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=kernel_size),
            nn.PReLU(channels)
        )

    @numba.jit()
    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    img = torch.randn(48, 48, 3)
    img = Image.fromarray(np.array(img, dtype=np.uint8))
    input = torchvision.transforms.ToTensor()(img).unsqueeze(0)
    net = O48Net()
    # output = net(input)
    # print(output.shape)
    conf, offset = net(input)
    print(conf.shape, offset.shape)
