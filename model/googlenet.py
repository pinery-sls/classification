'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
from utils.options import args

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        #* 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        #* 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        #* 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        #* 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        if args.dataset == 'Imagenet':
            self.pre_layers = nn.Sequential(
                # Nx3x224x224
                nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                # Nx64x112x112
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Nx64x56x56
                nn.Conv2d(64, 64, kernel_size=1),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                # Nx192x56x56
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                # Nx192x28x28
            )
        else:
            self.pre_layers = nn.Sequential(
                # Nx3x32x32
                nn.Conv2d(3, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(True),
                # Nx192x32x32
            )

        self.a3 = Inception(192,    64,      96, 128,    16,  32,     32)
        self.b3 = Inception(256,    128,    128, 192,    32,  96,     64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480,    192,     96, 208,    16,  48,     64)
        self.b4 = Inception(512,    160,    112, 224,    24,  64,     64)
        self.c4 = Inception(512,    128,    128, 256,    24,  64,     64)
        self.d4 = Inception(512,    112,    144, 288,    32,  64,     64)
        self.e4 = Inception(528,    256,    160, 320,    32, 128,    128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        if args.dataset == 'Imagenet':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        else:
            self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)

        # 192x32x32   28
        out = self.a3(out)
        # 256x32x32   
        out = self.b3(out)
        # 480x32x32
        out = self.maxpool(out)
        
        # 480x16x16  14
        out = self.a4(out)
        # 512x16x16
        out = self.b4(out)
        # 512x16x16
        out = self.c4(out)
        # 512x16x16
        out = self.d4(out)
        # 512x16x16
        out = self.e4(out)
        # 823x16x16
        out = self.maxpool(out)

        # 823x8x8     7
        out = self.a5(out)
        # 823x8x8
        out = self.b5(out)
        # 1024x8x8
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def googlenet_cifar():
    return GoogLeNet(num_classes=10)

def googlenet_imagent():
    return GoogLeNet(num_classes=1000)


def googlenet(cfg = None):
    if cfg == 'googlenet_cifar':
        return GoogLeNet(num_classes=10)
    elif cfg == 'googlenet_imagenet':
        return GoogLeNet(num_classes=1000)