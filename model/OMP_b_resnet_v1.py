import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)

class ResNetBasicblock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        return F.relu(residual + basicblock, inplace=True)

class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """
    def __init__(self, block, depth, num_classes, num_convs=5):
        super(CifarResNet, self).__init__()

        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.num_classes = num_classes
        self.num_convs = num_convs

        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        # self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)

        stage_2_conv1 = []
        stage_2_conv1.append(block(self.inplanes, 32, 2, DownsampleA(self.inplanes, 32*block.expansion, 2)))
        self.stage_2_conv1 = nn.Sequential(*stage_2_conv1)

        # initialize the middle convolution layers
        stage_2_conv2 = []
        for i in range(num_convs):
            stage_2_conv2.append(block(32 * block.expansion, 32))
        self.stage_2_conv2 = nn.Sequential(*stage_2_conv2)

        stage_2_conv3 = []
        stage_2_conv3.append(block(32 * block.expansion, 32))
        self.stage_2_conv3 = nn.Sequential(*stage_2_conv3)

        self.inplanes = 32 * block.expansion


        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _orthogonal_costr(self):
        total = 0
        for i in range(self.num_convs):
            for param in self.stage_2_conv2[i].conv_a.parameters():
                conv_i_a_param = param
            for param in self.stage_2_conv2[i].conv_b.parameters():
                conv_i_b_param = param

            for j in range(i+1, self.num_convs):
                for param in self.stage_2_conv2[j].conv_a.parameters():
                    conv_j_a_param = param
                for param in self.stage_2_conv2[j].conv_b.parameters():
                    conv_j_b_param = param
                
                inner_prod_1 = conv_i_a_param.mul(conv_j_a_param).sum()
                inner_prod_2 = conv_i_a_param.mul(conv_j_b_param).sum()
                inner_prod_3 = conv_i_b_param.mul(conv_j_a_param).sum()
                inner_prod_4 = conv_i_b_param.mul(conv_j_b_param).sum()
                total = total + inner_prod_1 * inner_prod_1 + inner_prod_2 * inner_prod_2 + inner_prod_3 * inner_prod_3 + inner_prod_4 * inner_prod_4
        
        return total

    def forward(self, x, forward_type):

        """
        :param forward_type:
            'all':    return the predictions of ALL mutually-orthogonal paths
            'random': return the prediction  of ONE RANDOM path
            number:   return the prediction  of the SELECTED path
        """

        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2_conv1(x)

        if forward_type == 'all':
            all_logits = []
            for idx in range(self.num_convs):
                output = self.stage_2_conv2[idx](x)
                output = self.stage_2_conv3(output)
                output = self.stage_3(output)
                output = self.avgpool(output)
                output = output.view(output.size(0), -1)
                logits = self.classifier(output)
                all_logits.append(logits)
            return None, all_logits
        
        elif forward_type == 'random':
            output = self.stage_2_conv2[torch.randint(self.num_convs,(1,))](x)
            output = self.stage_2_conv3(output)
            output = self.stage_3(output)
            output = self.avgpool(output)
            output = output.view(output.size(0), -1)
            logits = self.classifier(output)
            return None, logits
        
        else:
            output = self.stage_2_conv2[forward_type](x)
            output = self.stage_2_conv3(output)
            output = self.stage_3(output)
            output = self.avgpool(output)
            output = output.view(output.size(0), -1)
            logits = self.classifier(output)
            return None, logits


def resnet20(num_classes=10, num_convs=5):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes, num_convs)
    return model

