import numpy as np

from main import args


def cuda(args, tensor):
    return tensor.to(args.device)


def rpad(tensor, rspace, c):
    return np.pad(
        tensor, (0, rspace), 'constant', constant_values=c
    ).tolist()


def optimizer_params(model):
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.wd
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    return params
