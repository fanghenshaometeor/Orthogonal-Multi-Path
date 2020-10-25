import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, num_classifiers=10):
        super().__init__()
        self.features = features
        self.num_classifiers = num_classifiers

        # ---- initialize 10 classifiers
        clfs = []
        for i in range(num_classifiers):
            clfs.append(nn.Linear(512, num_classes, bias=False))
        self.classifiers = nn.Sequential(*clfs)

    # ---- orthogonality constraint
    def _orthogonal_costr(self):
        total = 0
        for i in range(self.num_classifiers):
            for param in self.classifiers[i].parameters():
                clf_i_param = param
            for j in range(i+1, self.num_classifiers):
                for param in self.classifiers[j].parameters():
                    clf_j_param = param
                inner_prod = clf_i_param.mul(clf_j_param).sum()
                total = total + inner_prod * inner_prod
        return total

    def forward(self, x, forward_type):

        """
        :param forward_type:
            'all':    return the predictions of ALL mutually-orthogonal paths
            'random': return the prediction  of ONE RANDOM path
            number:   return the prediction  of the SELECTED path
        """

        x = self.features(x)
        x = x.view(x.size(0), -1)

        if forward_type == 'all':
            all_logits = []
            for idx in range(self.num_classifiers):
                output = self.classifiers[idx](x)
                all_logits.append(output)
            return None, all_logits
        
        elif forward_type == 'random':
            return None, self.classifiers[torch.randint(self.num_classifiers,(1,))](x)
        
        else:
            return x, self.classifiers[forward_type](x)


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
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

def vgg11_bn(num_classes=10, num_classifiers=10):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes, num_classifiers)

def vgg13_bn(num_classes=10, num_classifiers=10):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes, num_classifiers)

def vgg16_bn(num_classes=10, num_classifiers=10):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes, num_classifiers)

def vgg19_bn(num_classes=10, num_classifiers=10):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes, num_classifiers)
