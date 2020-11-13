"""
Created on Mon Feb 24 2020

@author: fanghenshao

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms

import os
import ast
import copy
import time
import random
import argparse
import numpy as np

from utils import setup_seed
from attackers import pgd_attack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Training Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='./save/',help='file path for saving model')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='OMPc',help='architecture of OMP model, alternative value include OMPa, OMPb and OMPc')
parser.add_argument('--model',type=str,default='vgg16',help='model name')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=512,help='batch size for training (default: 256)')    
parser.add_argument('--epochs',type=int,default=200,help='number of epochs to train (default: 200)')
parser.add_argument('--gpu_id',type=str,default='0',help='gpu device index')
# -------- enable adversarial training --------
parser.add_argument('--adv_train',type=ast.literal_eval,dest='adv_train',help='enable the adversarial training')
parser.add_argument('--adv_delay',type=int,default=10,help='epochs delay for adversarial training')
# -------- hyper parameters -------
parser.add_argument('--lamb',type=float,default=0.1,help='regularization parameters')
parser.add_argument('--num_paths',type=int,default=10,help='number of orthogonal paths')
parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
args = parser.parse_args()

# ======== GPU device ========
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# -------- main function
def main():
    
    # ======== data set preprocess =============
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == 'STL10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
        trainset = datasets.STL10(root=args.data_dir, split='train', transform=transform_train, download=True)
        testset = datasets.STL10(root=args.data_dir, split='test', transform=transform_test, download=True)
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== initialize net
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
    elif args.model == 'modela':
        if args.arch == 'OMPc':
            from model.OMP_c_modela import ModelA
            net = ModelA(args.num_classes, args.num_paths).cuda()
        else:
            assert False, "Unsupported {}+{}".format(args.arch, args.model)
    else:
        assert False, "Unknown model : {}".format(args.model)
    
    if args.adv_train:
        args.model_path = args.model_dir+args.dataset+'-'+args.arch+'-'+args.model+'-lamb-'+str(args.lamb)+'-path-'+str(args.num_paths)+'-adv.pth'
    else:
        args.model_path = args.model_dir+args.dataset+'-'+args.arch+'-'+args.model+'-lamb-'+str(args.lamb)+'-path-'+str(args.num_paths)+'.pth'
    print('-------- MODEL INFORMATION --------')
    print("---- arch :      "+args.arch)
    print('---- model:      '+args.model)
    print('---- adv. train: '+str(args.adv_train))
    print('---- saved path: '+args.model_path)

    # ======== set criterions & optimizers
    criterion = nn.CrossEntropyLoss()
    if args.model == 'vgg11' or args.model == 'vgg13' or args.model == 'vgg16' or args.model == 'vgg19':
        args.epochs = 200
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)
    elif args.model == 'resnet20' or args.model == 'resnet32':
        args.epochs = 350
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250],gamma=0.1)
    elif args.model == 'modela':
        args.epochs = 200 
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)


    print('-------- START TRAINING --------')

    for epoch in range(args.epochs):

        start = time.time()
        # -------- train
        loss_tr, loss_ortho = train_epoch(net, trainloader, testloader, optimizer, criterion, epoch)

        # -------- validation
        if epoch % 20 == 0 or epoch == (args.epochs-1):
            corr_tr, corr_te = val(net, trainloader, testloader)

        scheduler.step()

        duration = time.time() - start


        # -------- save model
        if epoch % 20 == 0 or epoch == (args.epochs-1):
            checkpoint = {'state_dict': net.state_dict()}
            torch.save(checkpoint, args.model_path)

        if args.adv_train:
            print('Epoch %d/%d costs %fs:' % (epoch, args.epochs, duration))
            print('     train loss on each path: ')
            print('     ', loss_tr['clean'])
            print('     train adv. loss on each path: ')
            print('     ', loss_tr['adv'])
            print('     orthogonal loss = %f/%f'%(loss_ortho['clean'], loss_ortho['adv']))
        else:
            print('Epoch %d/%d costs %fs:' % (epoch, args.epochs, duration))
            print('     train loss on each classifier: ')
            print('     ', loss_tr['clean'])
            print('     orthogonal loss = %f'%loss_ortho['clean'])
        
        if epoch % 20 == 0 or epoch == (args.epochs-1):
            print('     train acc. on each path: ')
            print('     ', corr_tr/train_num)
            print('     test  acc. on each path: ')
            print('     ', corr_te/test_num)
    

# ======== train  model ========
def train_epoch(net, trainloader, testloader, optim, criterion, epoch):
    
    net.train()

    loss_tr, loss_ortho = {}, {}
    avg_loss_tr, avg_loss_tr_adv = np.zeros(args.num_paths), np.zeros(args.num_paths)
    avg_loss_ortho_clean, avg_loss_ortho_adv = 0.0, 0.0

    for batch_idx, (b_data, b_label) in enumerate(trainloader):
        
        # -------- move to gpu
        b_data, b_label = b_data.cuda(), b_label.cuda()

        # ------- forward anc compute loss
        total_loss = 0
        _, all_logits = net(b_data, 'all')
        for idx in range(args.num_paths):
            logits = all_logits[idx]
            loss = criterion(logits, b_label)
            avg_loss_tr[idx] = avg_loss_tr[idx] + loss.item()       # save the loss value
            total_loss = total_loss + 1/args.num_paths * loss       # sum the weighted loss for backward propagation
        
        # ------- compute the orthogonal constraint
        if args.num_paths > 1:
            loss_ortho_clean = net._orthogonal_costr()
            avg_loss_ortho_clean = avg_loss_ortho_clean + loss_ortho_clean.item()
        else:
            loss_ortho_clean = 0
        
        total_loss = total_loss + args.lamb*loss_ortho_clean
        
        if batch_idx == (len(trainloader)-1):
            avg_loss_tr = avg_loss_tr / len(trainloader)
            avg_loss_ortho_clean = avg_loss_ortho_clean / len(trainloader)

        # -------- backprop. & update
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        # -------- training with adversarial examples
        if args.adv_train and epoch > args.adv_delay:
            net.eval()
            perturbed_data, _ = pgd_attack(net, b_data, b_label, eps=0.013, alpha=0.01, iters=7)
            net.train()

            total_loss = 0
            _, all_logits = net(perturbed_data, 'all')
            for idx in range(args.num_paths):
                logits = all_logits[idx]
                loss_adv = criterion(logits, b_label)
                avg_loss_tr_adv[idx] = avg_loss_tr_adv[idx] + loss_adv.item()       # save the loss value
                total_loss = total_loss + 1/args.num_paths * loss_adv         # sum the weighted loss for backward propagation       
            
            # ------- compute the orthogonal constraint
            if args.num_paths > 1:
                loss_ortho_adv = net._orthogonal_costr()
                avg_loss_ortho_adv = avg_loss_ortho_adv + loss_ortho_adv.item()
            else:
                loss_ortho_adv = 0
            
            total_loss = total_loss + args.lamb*loss_ortho_adv

            if batch_idx == (len(trainloader)-1):
                avg_loss_tr_adv = avg_loss_tr_adv / len(trainloader)
                avg_loss_ortho_adv = avg_loss_ortho_adv / len(trainloader)

            optim.zero_grad()
            total_loss.backward()
            optim.step()

    loss_tr['clean'], loss_tr['adv'] = avg_loss_tr, avg_loss_tr_adv
    loss_ortho['clean'], loss_ortho['adv'] = avg_loss_ortho_clean, avg_loss_ortho_adv  


    return loss_tr, loss_ortho

# ======== evaluate model ========
def val(net, trainloader, testloader):
    
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


# ======== startpoint
if __name__ == '__main__':
    main()