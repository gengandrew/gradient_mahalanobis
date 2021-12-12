import os
import time
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
from utils import sample_estimator, get_gradient_Mahalanobis_scores


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


def get_gradient_mahalanobis_hyperparameters(model, in_dataset, name, batch_size=128, random_seed=1):
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
    sample_mean, precision = sample_estimator(model, trainloaderIn, num_classes, temperature=1)

    np.save('./trained_models/{in_dataset}/{name}/mean.npy'.format(in_dataset=in_dataset, name=name), sample_mean)
    np.save('./trained_models/{in_dataset}/{name}/precision.npy'.format(in_dataset=in_dataset, name=name), precision)
    # sample_mean = np.load('./mean.npy', allow_pickle=True).item()
    # precision = np.load('./precision.npy', allow_pickle=True).item()

    print('Training logistic regression model')
    batch_size = 10
    m = 500
    train_in = []
    train_in_label = []
    train_out = []
    val_in = []
    val_in_label = []
    val_out = []

    cnt = 0
    for data, target in testloaderIn:
        data = data.numpy()
        target = target.numpy()
        for x, y in zip(data, target):
            cnt += 1
            if cnt <= m:
                train_in.append(x)
                train_in_label.append(y)
            elif cnt <= 2*m:
                val_in.append(x)
                val_in_label.append(y)

            if cnt == 2*m:
                break
        if cnt == 2*m:
            break

    criterion = nn.CrossEntropyLoss().cuda()
    adv_noise = 0.05

    for i in range(int(m/batch_size) + 1):
        if i*batch_size >= m:
            break

        data = torch.tensor(train_in[i*batch_size:min((i+1)*batch_size, m)])
        target = torch.tensor(train_in_label[i*batch_size:min((i+1)*batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        train_out.extend(adv_data.cpu().numpy())

    for i in range(int(m/batch_size) + 1):
        if i*batch_size >= m:
            break

        data = torch.tensor(val_in[i*batch_size:min((i+1)*batch_size, m)])
        target = torch.tensor(val_in_label[i*batch_size:min((i+1)*batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        val_out.extend(adv_data.cpu().numpy())

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(train_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(train_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)

    train_lr_Mahalanobis = []
    total = 0
    for data_index in range(int(np.floor(train_lr_data.size(0) / batch_size))):
        data = train_lr_data[total : total + batch_size].cuda()
        total += batch_size
        scores = get_gradient_Mahalanobis_scores(data, model, num_classes, sample_mean, precision)
        train_lr_Mahalanobis.extend(scores)

    train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
    regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

    return sample_mean, precision, regressor
