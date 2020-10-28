"""
Created on Mon Mar 09 2020

@author: fanghenshao
"""

from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms

import os
import ast
import argparse

import numpy as np
import scipy.io as sio

from utils import setup_seed

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Test Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='OMPc',help='architecture of OMP model, alternative value include OMPa, OMPb and OMPc')
parser.add_argument('--model',type=str,default='vgg16',help='model architecture name')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')
parser.add_argument('--gpu_id',type=str,default='0',help='gpu device index')
# -------- hyper parameters -------
parser.add_argument('--num_paths',type=int,default=10,help='number of orthogonal paths')
parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
args = parser.parse_args()

# ======== GPU device ========
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# -------- main function
def main():
    
    # ======== data set preprocess =============
    # ======== mean-variance normalization is removed
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== load network ========
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    if args.model == 'vgg11':
        if args.arch == 'OMPc':
            from model.OMP_c_vgg import vgg11_bn
            net = vgg11_bn(args.num_classes, args.num_paths).cuda()
        else:
            assert False, "Unsupported {}+{}".format(args.arch, args.model)
    elif args.model == 'vgg13':
        if args.arch == 'OMPc':
            from model.OMP_c_vgg import vgg13_bn
            net = vgg13_bn(args.num_classes, args.num_paths).cuda()
        else:
            assert False, "Unsupported {}+{}".format(args.arch, args.model)
    elif args.model == 'vgg16':
        if args.arch == 'OMPa':
            from model.OMP_a_vgg import vgg16_bn
            net = vgg16_bn(args.num_classes, args.num_paths).cuda()
        elif args.arch == 'OMPb':
            from model.OMP_b_vgg import vgg16_bn
            net = vgg16_bn(args.num_classes, args.num_paths).cuda()
        elif args.arch == 'OMPc':
            from model.OMP_c_vgg import vgg16_bn
            net = vgg16_bn(args.num_classes, args.num_paths).cuda()
        else:
            assert False, "Unknown architecture : {}".format(args.arch)
    elif args.model == 'vgg19':
        if args.arch == 'OMPc':
            from model.OMP_c_vgg import vgg19_bn
            net = vgg19_bn(args.num_classes, args.num_paths).cuda()
        else:
            assert False, "Unsupported {}+{}".format(args.arch, args.model)
    elif args.model == 'resnet20':
        if args.arch == 'OMPa':
            from model.OMP_a_resnet_v1 import resnet20
            net = resnet20(args.num_classes, args.num_paths).cuda()
        elif args.arch == 'OMPb':
            from model.OMP_b_resnet_v1 import resnet20
            net = resnet20(args.num_classes, args.num_paths).cuda()
        elif args.arch == 'OMPc':
            from model.OMP_c_resnet_v1 import resnet20
            net = resnet20(args.num_classes, args.num_paths).cuda()
        else:
            assert False, "Unknown architecture : {}".format(args.arch)
    elif args.model == 'resnet32':
        if args.arch == 'OMPc':
            from model.OMP_c_resnet_v1 import resnet32
            net = resnet32(args.num_classes, args.num_paths).cuda()
        else:
            assert False, "Unsupported {}+{}".format(args.arch, args.model)
    else:
        assert False, "Unknown model : {}".format(args.model)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('-------- MODEL INFORMATION --------')
    print("---- arch :      "+args.arch)
    print('---- model:      '+args.model)
    print('---- saved path: '+args.model_path)



    print('-------- START TESTING --------')
    corr_tr, corr_te = evaluate(net, trainloader, testloader)
    print('Train acc. of each path:')
    print('     ', corr_tr/train_num)
    print('Test  acc. of each path:')
    print('     ', corr_te/test_num)

    return

# -------- test model ---------------
def evaluate(net, trainloader, testloader):
    
    net.eval()
    
    correct_train, correct_test = np.zeros(args.num_paths), np.zeros(args.num_paths)
    
    with torch.no_grad():
        
        # -------- compute the accs. of train, test set
        for test in testloader:
            images, labels = test
            images, labels = images.cuda(), labels.cuda()
            
            # ------- forward 
            _, all_logits = net(images, 'all')
            for idx in range(args.num_paths):
                logits = all_logits[idx]
                logits = logits.detach()
                _, pred = torch.max(logits.data, 1)
                correct_test[idx] += (pred == labels).sum().item()
 
        for train in trainloader:
            images, labels = train
            images, labels = images.cuda(), labels.cuda()
            
            _, all_logits = net(images, 'all')
            for idx in range(args.num_paths):
                logits = all_logits[idx]
                logits = logits.detach()
                _, pred = torch.max(logits.data, 1)
                correct_train[idx] += (pred == labels).sum().item()

    return correct_train, correct_test



# -------- start point
if __name__ == '__main__':
    main()