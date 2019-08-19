import torch.nn as nn


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),  # conv1
            nn.PReLU(10),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # pool
            nn.Conv2d(10, 16, 3, 1),  # conv2
            nn.PReLU(16),
            nn.Conv2d(16, 32, 3, 1),  # conv3
            nn.PReLU(32)
        )

        self.conv4_1 = nn.Conv2d(32, 1, 1, 1)  # conv4
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)  # conv5

    def forward(self, x):
        output = self.prelayer(x)
        # confidence = nn.functional.sigmoid(self.conv4_1(output))
        confidence = self.conv4_1(output).sigmoid()
        offset = self.conv4_2(output)

        return confidence, offset


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1),  # conv1
            nn.PReLU(28),
            nn.MaxPool2d(3, 2, ceil_mode=True),  # pool
            nn.Conv2d(28, 48, 3, 1),  # conv2
            nn.PReLU(48),
            nn.MaxPool2d(3, 2, ceil_mode=True),  # pool2
            nn.Conv2d(48, 64, 2, 2),  # conv3
            nn.PReLU(64)
            # nn.Linear(576, 128),
            # nn.PReLU(128)
        )
        self.conv4 = nn.Linear(64 * 2 * 2, 128)
        self.prelu4 = nn.PReLU()

        self.conv5_1 = nn.Linear(128, 1)
        self.conv5_2 = nn.Linear(128, 4)

    def forward(self, x):
        output = self.pre_layer(x)
        # print("1", output.shape)
        output = output.view(output.size(0), -1)
        # print('2', output.shape)
        output = self.conv4(output)
        output = self.prelu4(output)

        # label = nn.functional.sigmoid(self.conv5_1(output))
        label = self.conv5_1(output).sigmoid()
        offset = self.conv5_2(output)

        return label, offset


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelayer0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.PReLU(32),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.PReLU(64),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.PReLU(64),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(64, 128, 2, 1),
            nn.PReLU(128)
        )

        self.conv5 = nn.Linear(3 * 3 * 128, 256)
        self.prelu5 = nn.PReLU()

        self.conv6_1 = nn.Linear(256, 1)
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        output = self.prelayer0(x)
        output = output.view(x.size(0), -1)
        output = self.conv5(output)
        output = self.prelu5(output)

        label = self.conv6_1(output).sigmoid()
        offset = self.conv6_2(output)

        return label, offset
