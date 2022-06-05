import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, inplance, output_size, height, width, kernelsize_1=3, kernelsize_2=3):
        super(AutoEncoder, self).__init__()
        self.kernel_size1 = kernelsize_1
        self.kernel_size2 = kernelsize_2

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=inplance, out_channels=inplance // 4, kernel_size=kernelsize_1, bias=False
        )
        self.enc2 = nn.Conv2d(
            in_channels=inplance // 4, out_channels=inplance // 16, kernel_size=kernelsize_2, bias=False
        )
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=inplance // 16, out_channels=inplance//4, kernel_size=kernelsize_2, bias=False
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=inplance//4, out_channels=output_size, kernel_size=kernelsize_1, bias=False
        )
    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        x1 = F.relu(self.enc2(x1))
        x1 = F.relu(self.dec1(x1))
        x1 = F.relu(self.dec2(x1))

        return x1



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        
        self.autoencoder1 = AutoEncoder(inplance=64, output_size=64, height=64, width=64)
        self.autoencoder2 = AutoEncoder(inplance=512, output_size=512, height=6, width=6)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # out : [64, 64, 44, 44]

        x1 =self.autoencoder1(out)

        out = self.layer1(x1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # out : [64, 512, 6, 6]

        x1 = self.autoencoder2(out)

        out = F.avg_pool2d(x1, 4)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
