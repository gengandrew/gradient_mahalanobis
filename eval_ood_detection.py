import os
import time
import torch
import argparse
import numpy as np
import torchvision
import numpy as np
import torch.nn as nn
import models.resnet as rn
import models.densenet as dn
import models.wideresnet as wn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
from utils import get_Mahalanobis_score, get_gradient_Mahalanobis_scores
from find_mahalanobis_hyperparams import get_best_mahalanobis_hyperparams
from find_gradient_mahalanbis import get_gradient_mahalanobis_hyperparameters


parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--name', required=True, type=str, help='the name of the model trained')
parser.add_argument('--model-arch', default='resnet34', type=str, help='model architecture')
parser.add_argument('--gpu', default = '0', type = str, help='gpu index')
parser.add_argument('--method', default='gradient_mahalanobis', type=str, help='scoring function')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=50, type=int, help='mini-batch size')
parser.add_argument('--base-dir', default='evaluations', type=str, help='result directory')

parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int, help='depth of resnet')
parser.add_argument('--width', default=4, type=int, help='width of resnet')
parser.set_defaults(argument=True)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def get_msp_score(inputs, model, method_args):
    with torch.no_grad():
        outputs = model(inputs)
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores


def get_odin_score(inputs, model, method_args):
    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = Variable(inputs, requires_grad = True)
    outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper

    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_energy_score(inputs, model, method_args, temper=1):
    with torch.no_grad():
        logits = model(inputs)
        scores = temper * torch.logsumexp(logits / temper, dim=1)

    return scores


def get_mahalanobis_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    sample_mean = method_args['sample_mean']
    precision = method_args['precision']
    magnitude = method_args['magnitude']
    regressor = method_args['regressor']
    num_output = method_args['num_output']

    Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude)
    scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]

    return scores


def get_gradnorm_score(inputs, model, num_classes, temperature=1):
    scores = []
    for input in inputs:
        input = input.unsqueeze(0)
        uniform_target = torch.ones((input.shape[0], num_classes)).cuda()

        # Model forward with temperature scaling
        model.zero_grad()
        outputs = model(input)
        outputs = outputs / temperature

        # Back propagation for gradient
        logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
        loss = torch.mean(torch.sum(-uniform_target * logsoftmax(outputs), dim=-1))
        loss.backward()

        # Calculate the GradNorm of model
        if isinstance(model, nn.DataParallel):
            score = torch.sum(torch.abs(model.module.head.weight.grad.data)).cpu().item()
        else:
            score = torch.sum(torch.abs(model.head.weight.grad.data)).cpu().item()

        scores.append(score)
    
    return scores


def get_gradient_mahalanobis_score(inputs, model, method_args):
    regressor = method_args['regressor']
    gradient_Mahalanobis_scores = get_gradient_Mahalanobis_scores(inputs, model, method_args['num_classes'], method_args['sample_mean'], method_args['precision'])
    scores = -regressor.predict_proba(gradient_Mahalanobis_scores)[:, 1]

    return scores


def get_score(inputs, model, method, method_args):
    if method == "msp":
        scores = get_msp_score(inputs, model, method_args)
    elif method == "odin":
        scores = get_odin_score(inputs, model, method_args)
    elif method == "energy":
        scores = get_energy_score(inputs, model, method_args)
    elif method == "mahalanobis":
        scores = get_mahalanobis_score(inputs, model, method_args)
    elif method == "GradNorm":
        scores = get_gradnorm_score(inputs, model, method_args['num_classes'])
    elif method == 'gradient_mahalanobis':
        scores = get_gradient_mahalanobis_score(inputs, model, method_args)

    return scores


def eval_ood_detector(base_dir, in_dataset, out_datasets, batch_size, method, method_args, name, epochs):
    in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'nat')
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if in_dataset == "CIFAR-10":
        testset = torchvision.datasets.CIFAR10(root='../../GradNorm_OE/datasets/cifar10', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
        num_classes = 10
    elif in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='../../GradNorm_OE/datasets/cifar100', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
        num_classes = 100

    method_args['num_classes'] = num_classes

    if args.model_arch == 'resnet18':
        model = rn.resnet18(num_classes)
    elif args.model_arch == 'resnet34':
        model = rn.resnet34(num_classes)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(28, num_classes, widen_factor=20, dropRate=args.droprate)
    elif args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    checkpoint = torch.load("./trained_models/{in_dataset}/{name}/epoch_{epochs}.pth".format(in_dataset=in_dataset, name=name, epochs=epochs))
    model.load_state_dict(checkpoint)

    model.eval()
    model.cuda()

    if method == "mahalanobis":
        method_args = get_best_mahalanobis_hyperparams(model, name, in_dataset)
        temp_x = torch.rand(2,3,32,32)
        temp_x = Variable(temp_x).cuda()
        if isinstance(model, nn.DataParallel):
            temp_list = model.module.feature_list(temp_x)[1]
        else:
            temp_list = model.feature_list(temp_x)[1]

        num_output = len(temp_list)
        method_args['num_output'] = num_output
        method_args['num_classes'] = num_classes
    elif method == 'gradient_mahalanobis':
        sample_mean, precision, regressor = get_gradient_mahalanobis_hyperparameters(model, in_dataset, name)
        method_args['sample_mean'] = sample_mean
        method_args['precision'] = precision
        method_args['num_classes'] = num_classes
        method_args['regressor'] = regressor

    t0 = time.time()

    f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
    g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

    ########################################In-distribution###########################################
    print("Processing in-distribution images")
    N = len(testloaderIn.dataset)
    count = 0
    for j, (inputs, target) in enumerate(testloaderIn):
        inputs = inputs.cuda()
        target = target.cuda()

        scores = get_score(inputs, model, method, method_args)
        for score in scores:
            f1.write("{}\n".format(score))

        outputs = F.softmax(model(inputs)[:, :num_classes], dim=1)
        outputs = outputs.detach().cpu().numpy()
        preds = np.argmax(outputs, axis=1)
        confs = np.max(outputs, axis=1)

        for k in range(preds.shape[0]):
            g1.write("{} {} {}\n".format(target[k], preds[k], confs[k]))

        count += inputs.shape[0]
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
        t0 = time.time()

    f1.close()
    g1.close()

    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir, out_dataset)
        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')
        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        if out_dataset == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root="../../GradNorm_OE/datasets/ood_datasets/dtd/images",
                                        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True,
                                                     num_workers=2)
        elif out_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root="../../GradNorm_OE/datasets/ood_datasets/places365/test_subset",
                                        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True,
                                                     num_workers=2)
        else:
            testsetout = torchvision.datasets.ImageFolder("../../GradNorm_OE/datasets/ood_datasets/{}".format(out_dataset),
                                        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

        ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        count = 0
        for j, (inputs, target) in enumerate(testloaderOut):
            inputs = inputs.cuda()

            scores = get_score(inputs, model, method, method_args)
            for score in scores:
                f2.write("{}\n".format(score))

            count += inputs.shape[0]
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()
        f2.close()

    return


if __name__ == '__main__':
    method_args = dict()
    out_datasets = ['LSUN_resize', 'iSUN', 'dtd', 'places365']

    if args.method == 'msp':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs)
    elif args.method == 'energy':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs)
    elif args.method == 'GradNorm':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs)
    elif args.method == 'gradient_mahalanobis':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs)
    elif args.method == 'mahalanobis':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs)
    elif args.method == "odin":
        method_args['temperature'] = 1000.0
        if args.model_arch == 'densenet':
            if args.in_dataset == "CIFAR-10":
                method_args['magnitude'] = 0.0016
            elif args.in_dataset == "CIFAR-100":
                method_args['magnitude'] = 0.0012
            elif args.in_dataset == "SVHN":
                method_args['magnitude'] = 0.0006
        elif args.model_arch == 'wideresnet':
            if args.in_dataset == "CIFAR-10":
                method_args['magnitude'] = 0.0006
            elif args.in_dataset == "CIFAR-100":
                method_args['magnitude'] = 0.0012
            elif args.in_dataset == "SVHN":
                method_args['magnitude'] = 0.0002
        elif args.model_arch == 'resnet34':
            if args.in_dataset == "CIFAR-10":
                method_args['magnitude'] = 0.0006
            elif args.in_dataset == "CIFAR-100":
                method_args['magnitude'] = 0.0012
            elif args.in_dataset == "SVHN":
                method_args['magnitude'] = 0.0002
        else:
            assert False, 'Not supported model arch'

        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs)