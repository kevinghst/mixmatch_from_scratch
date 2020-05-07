import numpy as np
import datetime
import torch
import random

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mixup_op(input, l, idx):
    input_a, input_b = input, input[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    return mixed_input

def pad_for_word_mixup(input_ids, input_mask, num_tokens, idx):
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
            second = torch.tensor([1] * (big_count - small_count))
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