import os
import time
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
from gradient_mahalanbis_utils import sample_estimator


def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
        print('')


def get_gradient_mahalanobis_hyperparameters(model, in_dataset, batch_size=128, random_seed=1):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    if in_dataset == "CIFAR-10":
        trainset= torchvision.datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

        num_classes = 10
    elif in_dataset == "CIFAR-100":
        trainset= torchvision.datasets.CIFAR100('./datasets/cifar100', train=True, download=True, transform=transform)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

        num_classes = 100
    else:
        assert False, 'Not supported In-distribution dataset: {in_dataset}'.format(in_dataset=in_dataset)

    model.eval()
    # sample_mean, precision = sample_estimator(model, trainloaderIn, num_classes, temperature=1)

    # np.save('./mean.npy', sample_mean)
    # np.save('./precision.npy', precision)
    # exit()
    sample_mean = np.load('./mean.npy', allow_pickle=True).item()
    precision = np.load('./precision.npy', allow_pickle=True).item()

    return sample_mean, precision
