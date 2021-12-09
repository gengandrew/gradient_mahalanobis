import torch
import numpy as np
from tqdm import tqdm
import sklearn.covariance
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegressionCV


def extract_layered_gradients(model, chosen_layers):
    batch_GradNorm_dict = dict()
    for name, param in model.named_parameters():
        if name not in chosen_layers:
            continue
        
        batch_GradNorm_dict[name] = torch.flatten(param.grad).cpu().numpy()

    return batch_GradNorm_dict


def sample_estimator(model, train_loader, num_classes, temperature=1):
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    chosen_layers = [name for name, param in model.named_parameters() if 'bias' not in name and 'bn' not in name]
    chosen_layers = chosen_layers[len(chosen_layers)-1:len(chosen_layers)]
    
    model.eval()
    GradNorm_dict = {name: [] for name in chosen_layers}
    for i, (inputs, target) in enumerate(tqdm(train_loader)):
        inputs = Variable(inputs.cuda(), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)

        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs/temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
        loss.backward()

        batch_GradNorm_dict = extract_layered_gradients(model, chosen_layers)
        for name, gradient in batch_GradNorm_dict.items():
            GradNorm_dict[name].append(gradient)
    
    GradNorm_mean_dict = dict()
    GradNorm_precision_dict = dict()
    for name, gradient_list in GradNorm_dict.items():
        gradient_list = np.array(gradient_list)
        GradNorm_mean_dict[name] = np.mean(gradient_list, axis=0)
        
        normalized_gradient_list = np.array([(each-GradNorm_mean_dict[name]) for each in gradient_list])
        group_lasso.fit(normalized_gradient_list)
        GradNorm_precision_dict[name] = group_lasso.precision_
        print('Precision at {name} calculated with shape {shape}'.format(name=name, shape=GradNorm_precision_dict[name].shape))

    return GradNorm_mean_dict, GradNorm_precision_dict


def get_gradient_Mahalanobis_scores(inputs, model, num_classes, sample_mean, precision, temperature=1):
    chosen_layers = [name for name, param in model.named_parameters() if 'bias' not in name and 'bn' not in name]
    chosen_layers = chosen_layers[len(chosen_layers)-1:len(chosen_layers)]
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    Mahalanobis_scores = []

    for input in inputs:
        input = torch.stack([input]).cuda()
        input = Variable(input, requires_grad=True)

        model.zero_grad()
        outputs = model(input)

        targets = torch.ones((input.shape[0], num_classes)).cuda()
        outputs = outputs/temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
        loss.backward()

        GradNorm_list = extract_layered_gradients(model, chosen_layers)

        Mahalanobis_score = []
        for name, gradient in GradNorm_list.items():
            normalized_gradient = torch.stack([torch.Tensor(gradient - sample_mean[name])])
            current_precision = torch.Tensor(precision[name])
            term_gau = -0.5*torch.mm(torch.mm(normalized_gradient, current_precision), normalized_gradient.t()).diag()
            Mahalanobis_score.append(term_gau.item())
        
        # normalized_gradient_list = np.array([(each-GradNorm_mean_dict[name]) for each in gradient_list])
        # normalized_GradNorm = torch.stack([torch.Tensor(GradNorm_list-sample_mean)])
        # term_gau = -0.5*torch.mm(torch.mm(normalized_GradNorm, precision), normalized_GradNorm.t()).diag()
        # Mahalanobis_score = term_gau.item()

        # Mahalanobis_scores.append(Mahalanobis_score)
        Mahalanobis_scores.append(Mahalanobis_score[0])

    return Mahalanobis_scores