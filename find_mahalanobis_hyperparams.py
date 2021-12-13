import os
import time
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
from utils import metric, sample_estimator, get_Mahalanobis_score


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


def get_mahalanobis_regressor_from_magnitude(model, in_dataset, magnitude, batch_size=10):
    print('Finding Mahalanobis regressor')

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

    # set information about feature extaction
    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x).cuda()
    if isinstance(model, nn.DataParallel):
        temp_list = model.module.feature_list(temp_x)[1]
    else:
        temp_list = model.feature_list(temp_x)[1]

    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    sample_mean, precision = sample_estimator(model, num_classes, feature_list, trainloaderIn)

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
        Mahalanobis_scores = get_Mahalanobis_score(data, model, num_classes, sample_mean, precision, num_output, magnitude)
        train_lr_Mahalanobis.extend(Mahalanobis_scores)

    train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
    regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

    print('Logistic Regressor params:', regressor.coef_, regressor.intercept_)
    return sample_mean, precision, regressor


def tune_mahalanobis_hyperparams(model, name, in_dataset, batch_size=10):
    print('Tuning hyper-parameters...')
    stypes = ['mahalanobis']

    save_dir = os.path.join('./mahalanobis_hyperparams/', in_dataset, name, 'tmp')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
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

    # set information about feature extaction
    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x).cuda()
    if isinstance(model, nn.DataParallel):
        temp_list = model.module.feature_list(temp_x)[1]
    else:
        temp_list = model.feature_list(temp_x)[1]

    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = sample_estimator(model, num_classes, feature_list, trainloaderIn)

    print('train logistic regression model')
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

    print('In', len(train_in), len(val_in))

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

    print('Out', len(train_out),len(val_out))

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(train_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(train_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)

    best_fpr = 1.1
    best_magnitude = 0.0

    for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
        train_lr_Mahalanobis = []
        total = 0
        for data_index in range(int(np.floor(train_lr_data.size(0) / batch_size))):
            data = train_lr_data[total : total + batch_size].cuda()
            total += batch_size
            Mahalanobis_scores = get_Mahalanobis_score(data, model, num_classes, sample_mean, precision, num_output, magnitude)
            train_lr_Mahalanobis.extend(Mahalanobis_scores)

        train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
        regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

        print('Logistic Regressor params:', regressor.coef_, regressor.intercept_)

        t0 = time.time()
        f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
        f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')

        print("Processing in-distribution images")

        count = 0
        for i in range(int(m/batch_size) + 1):
            if i * batch_size >= m:
                break
            images = torch.tensor(val_in[i * batch_size : min((i+1) * batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]
            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)
            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                f1.write("{}\n".format(-confidence_scores[k]))

            count += batch_size
            t0 = time.time()

        t0 = time.time()
        print("Processing out-of-distribution images")
        count = 0

        for i in range(int(m/batch_size) + 1):
            if i * batch_size >= m:
                break
            images = torch.tensor(val_out[i * batch_size : min((i+1) * batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                f2.write("{}\n".format(-confidence_scores[k]))

            count += batch_size
            t0 = time.time()

        f1.close()
        f2.close()

        results = metric(save_dir, stypes)
        print_results(results, stypes)
        fpr = results['mahalanobis']['FPR']
        if fpr < best_fpr:
            best_fpr = fpr
            best_magnitude = magnitude
            best_regressor = regressor

    print('Best Logistic Regressor params:', best_regressor.coef_, best_regressor.intercept_)
    print('Best magnitude', best_magnitude)

    return sample_mean, precision, best_regressor, best_magnitude


def get_best_mahalanobis_hyperparams(model, name, in_dataset, batch_size=10, random_seed=1):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    sample_mean, precision, best_regressor, best_magnitude = tune_mahalanobis_hyperparams(model, name, in_dataset, batch_size)
    save_dir = os.path.join('./trained_models/{in_dataset}/{name}/mahalanobis_hyperparams/'.format(in_dataset=in_dataset, name=name))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'results'), np.array([sample_mean, precision, best_regressor.coef_, best_regressor.intercept_, best_magnitude]))

    # Store Mahalanobis hyperparams inside dictionary
    method_args = dict()
    method_args['sample_mean'] = sample_mean
    method_args['precision'] = precision
    method_args['magnitude'] = best_magnitude
    method_args['regressor'] = best_regressor

    return method_args
