import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        if stride > 1:
            self.residual_function = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
            )
        else:
            self.residual_function = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
            )

        #shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, output_padding=1, stride=stride, bias=False)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class DeResNet(nn.Module):

    def __init__(self, block, num_block, in_channels=512, num_classes=10):
        super().__init__()

        self.in_channels = in_channels
        
        self.deconv1 = self._make_layer(block, 256, num_block[2], 2)
        self.deconv2 = self._make_layer(block, 128, num_block[1], 2)
        self.deconv3 = self._make_layer(block, 64, num_block[0], 2)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid())

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.deconv1(x)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)

        return output


def deresnet18(num_classes):
    """ return a ResNet 18 object
    """
    return DeResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def deresnet34(num_classes):
    """ return a ResNet 34 object
    """
    return DeResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def deresnet50(num_classes):
    """ return a ResNet 50 object
    """
    return DeResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def deresnet101(num_classes):
    """ return a ResNet 101 object
    """
    return DeResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def deresnet152(num_classes):
    """ return a ResNet 152 object
    """
    return DeResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)