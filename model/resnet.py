import torch.nn as nn
from utils.options import args

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, self.expansion * planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if args.dataset == 'Imagenet':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if args.dataset == 'Imagenet':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        else:
            self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Nx3x32x32
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Nx64x16x16
        out = self.layer1(out)
        # Nx64x16x16
        out = self.layer2(out)
        # Nx128x8x8
        out = self.layer3(out)
        # Nx256x4x4
        out = self.layer4(out)
        # Nx512x2x2
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnet(cfg = None):
    if cfg == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2])
    elif cfg == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3])
    elif cfg == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3])
    elif cfg == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3])
    elif cfg == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=10)
    if cfg == 'resnet18_cifar':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    elif cfg == 'resnet34_cifar':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10)
    elif cfg == 'resnet50_cifar':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
    elif cfg == 'resnet101_cifar':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=10)
    elif cfg == 'resnet152_cifar':
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=10)

def resnet18_cifar():
    return ResNet(BasicBlock, [2,2,2,2], num_classes=10)

def resnet34_cifar():
    return ResNet(BasicBlock, [3,4,6,3], num_classes=10)

def resnet50_cifar():
    return ResNet(Bottleneck, [3,4,6,3], num_classes=10)

def resnet101_cifar():
    return ResNet(Bottleneck, [3,4,23,3], num_classes=10)

def resnet152_cifar():
    return ResNet(Bottleneck, [3,8,36,3], num_classes=10)


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(Bottleneck, [3,4,6,3])

def resnet101():
    return ResNet(Bottleneck, [3,4,23,3])

def resnet152():
    return ResNet(Bottleneck, [3,8,36,3])