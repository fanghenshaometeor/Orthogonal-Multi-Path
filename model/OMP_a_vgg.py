import torch
import torch.nn as nn
import torch.nn.functional as F

# cfg = {
#     'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
#     'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
#     'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
#     'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# }

# CANCEL the 1st conv
cfg = {
    'A' : [    'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, num_convs=10):
        super().__init__()
        self.features = features
        self.num_convs = num_convs

        # ---- initialize 10 1st convolution layers
        convs = []
        for i in range(num_convs):
            convs.append(nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False))
        self.convs = nn.Sequential(*convs)
        self.bn = nn.BatchNorm2d(64)
        

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

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

        if forward_type == 'all':
            all_logits = []
            for idx in range(self.num_convs):
                output = self.convs[idx](x)
                output = F.relu(self.bn(output), inplace=True)
                output = self.features(output)
                output = output.view(output.size(0),-1)
                logits = self.classifier(output)
                all_logits.append(logits)
            return None, all_logits
        elif forward_type == 'random':
            output = self.convs[torch.randint(self.num_convs,(1,))](x)
            output = F.relu(self.bn(output), inplace=True)
            output = self.features(output)
            output = output.view(output.size(0),-1)
            logits = self.classifier(output)
            return None, logits
        else:
            output = self.convs[forward_type](x)
            output = F.relu(self.bn(output), inplace=True)
            output = self.features(output)
            output = output.view(output.size(0),-1)
            logits = self.classifier(output)
            return None, logits

def make_layers(cfg, batch_norm=False):
    layers = []

    # input_channel = 3
    input_channel = 64
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]#, bias=False)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)



def vgg16_bn(num_classes=10, num_convs=10):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes, num_convs)


