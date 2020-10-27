import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, cfg, num_classes=10, num_convs=10):
        super().__init__()
        self.num_classes = num_classes
        self.num_convs = num_convs

        self.features_front, self.convs, self.features_back = self.make_layers(cfg, self.num_convs, True)
        

        self.classifier = nn.Sequential(
            nn.Linear(512, self.num_classes)
        )

    def make_layers(self, config, num_convs=10, batch_norm=True):
        layers_front, layers_ortho, layers_back = [], [], []

        # split the cfg at the 1st value after the 1st M
        cfg_front = config[0:3]
        cfg_back = config[4:]

        # initialize the orthogonal conv. layers
        for i in range(num_convs):
            layers_ortho.append(nn.Conv2d(64,128,kernel_size=3,padding=1,bias=False))

        # initialize the front layers
        input_channel = 3
        for l in cfg_front:
            if l == 'M':
                layers_front += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers_front += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]#, bias=False)]

            if batch_norm:
                layers_front += [nn.BatchNorm2d(l)]
            
            layers_front += [nn.ReLU(inplace=True)]
            input_channel = l

        # initialize the back layers        
        layers_back += [nn.BatchNorm2d(128)]
        layers_back += [nn.ReLU(inplace=True)]
        input_channel = 128
        for l in cfg_back:
            if l == 'M':
                layers_back += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers_back += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]#, bias=False)]

            if batch_norm:
                layers_back += [nn.BatchNorm2d(l)]
            
            layers_back += [nn.ReLU(inplace=True)]
            input_channel = l

        return nn.Sequential(*layers_front), nn.Sequential(*layers_ortho), nn.Sequential(*layers_back)

    # ---- orthogonality constraint
    def _orthogonal_costr(self):
        total = 0
        for i in range(self.num_convs):
            for param in self.convs[i].parameters():
                conv_i_param = param
            for j in range(i+1, self.num_convs):
                for param in self.convs[j].parameters():
                    conv_j_param = param
                inner_prod = conv_i_param.mul(conv_j_param).sum()
                total = total + inner_prod * inner_prod
        return total

    def forward(self, x, forward_type):

        """
        :param forward_type:
            'all':    return the predictions of ALL mutually-orthogonal paths
            'random': return the prediction  of ONE RANDOM path
            number:   return the prediction  of the SELECTED path
        """

        x = self.features_front(x)

        if forward_type == 'all':
            all_logits = []
            for idx in range(self.num_convs):
                output = self.convs[idx](x)
                output = self.features_back(output)
                output = output.view(output.size(0),-1)
                logits = self.classifier(output)
                all_logits.append(logits)
            return None, all_logits
        elif forward_type == 'random':
            output = self.convs[torch.randint(self.num_convs,(1,))](x)
            output = self.features_back(output)
            output = output.view(output.size(0),-1)
            logits = self.classifier(output)
            return None, logits
        else:
            output = self.convs[forward_type](x)
            output = self.features_back(output)
            output = output.view(output.size(0),-1)
            logits = self.classifier(output)
            return None, logits



def vgg16_bn(num_classes=10, num_convs=10):
    return VGG(cfg['D'], num_classes, num_convs)

