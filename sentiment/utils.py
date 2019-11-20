import numpy as np


def cuda(args, tensor):
    return tensor.to(args.device)


def rpad(tensor, rspace, c):
    return np.pad(
        tensor, (0, rspace), 'constant', constant_values=c
    ).tolist()
