import torch.nn as nn


class DBA_Block(nn.Module):

    def __init__(self, inChannals, outChannals):
        super(DBA_Block, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, outChannals, kernel_size=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        resudial = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += resudial
        out = self.relu(out)
        return out



class CPSR_s(nn.Module):
    """CPSR_subnetwork(4x)"""
    def __init__(self, inC, outC):
        super(CPSR_s, self).__init__()
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=5, padding=2, padding_mode='reflect', stride=1)  # 输入Y单通道图片，input channel = 1
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.resBlock = self._makeLayer_(DBA_Block, 128, 128, 6)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, outC, kernel_size=5, stride=1, padding=2, padding_mode='reflect')       # 输出Y单通道图片，output channel = 1

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        layers = []
        layers.append(block(inChannals, outChannals))
        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        residual = x

        x = self.conv2(x)
        x = self.relu(x)
        x = self.resBlock(x)
        x = self.conv3(x)
        x = self.bn3(x)
        out = residual + x
        out = self.relu(out)

        out = self.conv4(out)

        return out