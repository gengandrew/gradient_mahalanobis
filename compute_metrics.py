import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--name', required=True, type=str, help='neural network name and training set')
parser.add_argument('--method', default='msp', type=str, help='ood detection method')
parser.add_argument('--base-dir', default='evaluations', type=str, help='result directory')
parser.set_defaults(argument=True)
args = parser.parse_args()

np.random.seed(1)


def calculate_metrics(known, novel, method):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results


def get_curve(known, novel, method):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + in_dataset)
    print('out_distribution: '+ out_dataset)
    print('Model Name: ' + name)
    print('')

    print(' OOD detection method: ' + method)
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print('')


def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results


def compute_ood_performance(base_dir, in_dataset, out_datasets, method, name):
    print('OOD Detection Performance')

    known = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name), delimiter='\n')
    known_sorted = np.sort(known)
    num_k = known.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_sorted[round(0.05 * num_k)]

    all_results = []
    total = 0.0
    for out_dataset in out_datasets:
        novel = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/{out_dataset}/out_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset), delimiter='\n')
        total += novel.shape[0]
        results = calculate_metrics(known, novel, method)
        all_results.append(results)

    avg_results = compute_average_results(all_results)
    print_results(avg_results, in_dataset, "All", name, method)
    return


def compute_in_performance(base_dir, in_dataset, method, name):
    known_nat = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name), delimiter='\n')
    known_nat_sorted = np.sort(known_nat)
    num_k = known_nat.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_nat_sorted[round(0.05 * num_k)]

    known_nat_label = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_labels.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name))
    nat_in_cond = (known_nat>threshold).astype(np.float32)
    nat_correct = (known_nat_label[:,0] == known_nat_label[:,1]).astype(np.float32)
    known_nat_acc = np.mean(nat_correct)
    known_nat_fnr = np.mean((1.0 - nat_in_cond))
    known_nat_eteacc = np.mean(nat_correct * nat_in_cond)

    print('In-distribution performance:')
    print('FNR: {fnr:6.2f}, Acc: {acc:6.2f}, End-to-end Acc: {eteacc:6.2f}'.format(fnr=known_nat_fnr*100,acc=known_nat_acc*100,eteacc=known_nat_eteacc*100))
    return


if __name__ == '__main__':

    if args.in_dataset == "CIFAR-10" or args.in_dataset == "CIFAR-100":
        # out_datasets = ['iSUN']
        # out_datasets = ['LSUN_resize']
        # out_datasets = ['places365']
        # out_datasets = ['dtd']
        out_datasets = ['LSUN_resize', 'iSUN', 'dtd', 'places365']
    elif args.in_dataset == "SVHN":
        out_datasets = ['LSUN_C', 'LSUN_resize', 'iSUN', 'dtd', 'places365', 'CIFAR-10']

    compute_ood_performance(args.base_dir, args.in_dataset, out_datasets, args.method, args.name)
    compute_in_performance(args.base_dir, args.in_dataset, args.method, args.name)
