import numpy as np
import datetime
import torch
import random
import pdb
import math

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_ece(true, pred, conf):
    model = self.model

    p = conf.argsort()
    pred = pred[p]
    true = true[p]
    conf = conf[p]

    n = len(conf)
    bucket_size = 100
    buckets = int(n/bucket_size)

    pred = np.array_split(pred, buckets)
    true = np.array_split(true, buckets)
    conf = np.array_split(conf, buckets)

    ece = 0

    for i, conf_bucket in enumerate(conf):
        pred_bucket = pred[i]
        true_bucket = true[i]

        b_acc = np.sum(pred_bucket == true_bucket)
        b_conf = np.sum(conf_bucket)

        ece += (buckets/n) * (1/buckets * b_acc - 1/buckets * b_conf)**2
        
    pdb.set_trace()
    return math.sqrt(ece)

def mixup_op(input, l, idx):
    input_a, input_b = input, input[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    return mixed_input

def pad_for_word_mixup(input_ids, input_mask, num_tokens, idx):
    batch_size = input_ids.size(0)
    c_input_ids = input_ids.clone()

    for i in range(0, batch_size):
        j = idx[i]
        i_count = int(num_tokens[i])
        j_count = int(num_tokens[j])

        if i_count < j_count:
            small = i
            big = j
            small_count = i_count
            big_count = j_count
            small_ids = input_ids
            big_ids = c_input_ids
        elif i_count > j_count:
            small = j
            big = i
            small_count = j_count
            big_count = i_count
            small_ids = c_input_ids
            big_ids = input_ids

        if i_count != j_count:
            first = small_ids[small][0:small_count-1]
            second = torch.tensor([1] * (big_count - small_count)).cuda()
            third = big_ids[big][big_count-1:128]
            combined = torch.cat((first, second, third), 0)
            small_ids[small] = combined
            if i_count < j_count:
                input_mask[i] = input_mask[j]

    return input_ids, c_input_ids


# Function to calculate the accuracy of our predictions vs labels
def bin_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def multi_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

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

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)