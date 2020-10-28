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

from utils import setup_seed

import numpy as np

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Attack Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='OMPc',help='architecture of OMP model, alternative value include OMPa, OMPb and OMPc')
parser.add_argument('--model',type=str,default='vgg16',help='model architecture name')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=1024,help='batch size for training (default: 256)')
parser.add_argument('--gpu_id',type=str,default='0',help='gpu device index')
# -------- save adv. images --------
parser.add_argument('--save_adv_img',type=ast.literal_eval,dest='save_adv_img',help='save adversarial examples')
# -------- hyper parameters -------
parser.add_argument('--num_paths',type=int,default=10,help='number of orthogonal paths')
parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
parser.add_argument('--selected_path',type=int,default=0,help='selected path to be attacked')
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


    print('-------- START ATTACKING --------')
    print('---- selected path: '+str(args.selected_path))

    acc_fgsm, acc_pgd = np.zeros([10,12]), np.zeros([10,12])

    print('-------- START FGSM ATTACK...')
    fgsm_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
    for i,eps in enumerate(fgsm_epsilons):
        corr_te_fgsm, _, _ = attack(net, args.selected_path, testloader, eps, "FGSM")
        acc_fgsm[:,i] = corr_te_fgsm/float(test_num)
    print("acc. of other paths on the FGSM adversarial examples from the selected path:")
    print(acc_fgsm)

    print('-------- START PGD ATTACK...')
    pgd_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
    for i,eps in enumerate(pgd_epsilons):
        corr_te_pgd, _, _ = attack(net, args.selected_path, testloader, eps, "PGD")
        acc_pgd[:,i] = corr_te_pgd/float(test_num)
    print("acc. of other paths on the PGD  adversarial examples from the selected path:")
    print(acc_pgd)

    return

# -------- FGSM attack --------
def fgsm_attack(net, idx, image, label, epsilon):
    image.requires_grad = True

    _, logits = net(image, idx)
    loss = F.cross_entropy(logits, label)
    net.zero_grad()
    loss.backward()

    # collect data grad
    perturbed_image = image + epsilon*image.grad.data.sign()
    # clip the perturbed image into [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, None,

# -------- PGD attack --------
def pgd_attack(net, idx, image, label, eps, alpha=0.01, iters=7, random_start=True, d_min=0, d_max=1):

    perturbed_image = image.clone()
    perturbed_image.requires_grad = True

    image_max = image + eps
    image_min = image - eps
    image_max.clamp_(d_min, d_max)
    image_min.clamp_(d_min, d_max)

    if random_start:
        with torch.no_grad():
            perturbed_image.data = image + perturbed_image.uniform_(-1*eps, eps)
            perturbed_image.data.clamp_(d_min, d_max)
    
    for index in range(iters):
        net.zero_grad()
        _, logits = net(perturbed_image, idx)

        loss = F.cross_entropy(logits, label)
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
        
        loss.backward()
        data_grad = perturbed_image.grad.data

        with torch.no_grad():
            perturbed_image.data += alpha * torch.sign(data_grad)
            perturbed_image.data = torch.max(torch.min(perturbed_image, image_max), image_min)
    perturbed_image.requires_grad = False
    
    return perturbed_image, None

# -------- attack model --------
def attack(net, idx, testloader, epsilon, attackType):

    # correct = 0
    correct_total = np.zeros(10)

    net.eval()

    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        if attackType == "FGSM":
            perturbed_image, _ = fgsm_attack(net, idx, image, label, epsilon)
        elif attackType == "PGD":
            perturbed_image, _ = pgd_attack(net, idx, image, label, epsilon)

        if args.save_adv_img:
            adv_examples = perturbed_image[0:5,:,:,:].squeeze().detach().cpu().numpy()
            np.save('./log/%s-%.3f-vgg-adv'%(attackType,epsilon), adv_examples)

        # re-classify
        for re_clf_idx in range(args.num_paths):
            _, logits = net(perturbed_image, re_clf_idx)
            logits = logits.detach()
            _, pred = torch.max(logits.data, 1)
            correct_total[re_clf_idx] += (pred==label).sum().item()

    
    return correct_total, None, None


# -------- start point
if __name__ == '__main__':
    main()