import os
import abc
import sys
import copy
import time
import torch
import shutil
import argparse
import numpy as np
import torchvision
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import models.resnet as rn
import torch.nn.init as init
import models.densenet as dn
import models.wideresnet as wn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# nohup python -u train.py --name=vanilla_resnet34 --model-arch=resnet34 --in-dataset=CIFAR-10 --gpu=$1 >> ./vanilla_resnet34.out
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 & CIFAR-100 Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='resnet34', type=str, help='model architecture')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 0.0005)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: True)')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--name', required=True, type=str, help='name of experiment')

parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='To not use bottleneck block')
parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if epoch >= (0.5*args.epochs):
        lr *= 0.1
    
    if epoch >= (0.75*args.epochs):
        lr *= 0.1
    
    if epoch >= (0.9*args.epochs):
        lr *= 0.1
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(args, model, epoch):
    directory = "./trained_models/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = directory + 'epoch_{ep}.pth'.format(ep=epoch)
    torch.save(model.state_dict(), filename)

    print("{model_name} model saved at {file_name}".format(model_name=args.name, file_name=filename))


def validate(args, val_loader, model, criterion, num_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # Compute output
        output = model(input)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 75 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print('---------------> Accuracy {top1.avg:.3f} <---------------'.format(top1=top1))
    return top1.avg


def train(args, train_loader, model, criterion, optimizer, num_classes, epoch):
    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()

    # Switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        # Model forward propagation
        output = model(input)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        nat_prec1 = accuracy(output.data, target, topk=(1,))[0]
        nat_losses.update(loss.data, input.size(0))
        nat_top1.update(nat_prec1, input.size(0))

        # Gradient Descent Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print loss of the given the epoch and iteration
        if i % 75 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=nat_losses, top1=nat_top1))


def main(args):
    # Setting up augments for training data set
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
    
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    # Setting up testing dataset
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.in_dataset == "CIFAR-10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        num_classes = 100

    # Create the model
    if args.model_arch == 'resnet18':
        model = rn.resnet18(num_classes)
    elif args.model_arch == 'resnet34':
        model = rn.resnet34(num_classes)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(40, num_classes, widen_factor=4, dropRate=args.droprate)
    elif args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce, bottleneck=args.bottleneck, dropRate=args.droprate)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    model = model.cuda()
    cudnn.benchmark = True
    if ',' in args.gpu:
        model = nn.DataParallel(model)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    for epoch in range(0, args.epochs):
        # Update the step-wise learning rate
        adjust_learning_rate(args, optimizer, epoch)

        # Train for one epoch on the training set
        train(args, train_loader, model, criterion, optimizer, num_classes, epoch)

        # Evaluate on validation set
        validate(args, val_loader, model, criterion, num_classes)

    # Save final model
    save_checkpoint(args, model, epoch+1)


if __name__ == '__main__':
    # Grabbing cli arguments and printing results
    args = parser.parse_args()
    print_args = '*'*45
    for key,value in args._get_kwargs():
        print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

    print_args = print_args + '\n' + '*'*45
    print(print_args)

    # Creating checkpoint directory
    directory = "./trained_models/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Saving config args to checkpoint directory
    save_state_file = os.path.join(directory, 'args.txt')
    fw = open(save_state_file, 'w')
    print(print_args, file=fw)
    fw.close()

    # Setting up gpu parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Running main training method
    main(args)
