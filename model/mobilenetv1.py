import torch.nn as nn
import torch.nn.functional as F
from utils.options import args

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.conv1 = conv_bn(3, 32, 2)

        self.features = nn.Sequential(
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        #self.classifier = nn.Linear(1024, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        # Nx3x32x32
        x = self.conv1(x)
        # Nx32x16x16
        x = self.features(x)
        # Nx1024x1x1
        if args.dataset == "Imagenet":
            x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

def mobilenetv1(cfg = None):
    if cfg == 'mobilenetv1_cifar':
        return MobileNetV1(num_classes=10)
    elif cfg == 'mobilenetv1_imagenet':
        return MobileNetV1()
