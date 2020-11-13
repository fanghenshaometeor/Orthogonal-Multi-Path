import torch
import torch.nn as nn


class ModelA(nn.Module):
    def __init__(self, nclass=10, num_classifiers=10):
        super(ModelA, self).__init__()
        nchannel = 32
        self.num_classifiers = num_classifiers
        self.features = nn.Sequential(
                # 3x96x96
                nn.Conv2d(3, nchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # chx48x48
                nn.Conv2d(nchannel, 2*nchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(2*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 2chx24x24
                nn.Conv2d(2*nchannel, 4*nchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(4*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 4chx12x12
                nn.Conv2d(4*nchannel, 8*nchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(8*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 8chx6x6
                nn.Conv2d(8*nchannel, 8*nchannel, kernel_size=3, padding=0),
                nn.BatchNorm2d(8*nchannel),
                nn.ReLU(),
                # 8chx4x4
                nn.Conv2d(8*nchannel, 16*nchannel, kernel_size=3, padding=0),
                nn.BatchNorm2d(16*nchannel),
                nn.ReLU(),
                # 8chx2x2
                nn.AvgPool2d(kernel_size=2, stride=2)
                )
        
        # ---- initialize 10 classifiers
        clfs = []
        for i in range(num_classifiers):
            clfs.append(nn.Linear(16*nchannel, nclass, bias=False))
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

    def forward(self, input, forward_type):

        """
        :param forward_type:
            'all':    return the predictions of ALL mutually-orthogonal paths
            'random': return the prediction  of ONE RANDOM path
            number:   return the prediction  of the SELECTED path
        """

        out = self.features(input)
        out = out.view(out.size(0), -1)

        if forward_type == 'all':
            all_logits = []
            for idx in range(self.num_classifiers):
                output = self.classifiers[idx](out)
                all_logits.append(output)
            return None, all_logits
        
        elif forward_type == 'random':
            return None, self.classifiers[torch.randint(self.num_classifiers,(1,))](out)
        
        else:
            return None, self.classifiers[forward_type](out)

        # out = self.classifier(out)
        # return out
