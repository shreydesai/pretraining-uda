import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--device',
    type=int,
    default=0,
    help='cuda device (if available)',
)
parser.add_argument(
    '--ckpt',
    type=str,
    default='temp.pt',
    help='model checkpoint path',
)
parser.add_argument(
    '--src',
    type=str,
    default='apps',
    help='source domain dataset',
)
parser.add_argument(
    '--trg',
    type=str,
    default='beauty',
    help='target domain dataset',
)
parser.add_argument(
    '--model',
    type=str,
    default='bert-base-uncased',
    help='gpt2, bert-base-uncased, roberta-base, xlnet-base-uncased',
)
parser.add_argument(
    '--seq_len',
    type=int,
    default=256,
    help='maximum sequence length (<512)',
)
parser.add_argument(
    '--epochs',
    type=int,
    default=5,
    help='number of training epochs',
)
parser.add_argument(
    '--bs',
    type=int,
    default=8,
    help='batch size',
)
parser.add_argument(
    '--lr',
    type=float,
    default=2e-5,
    help='learning rate',
)
parser.add_argument(
    '--wd',
    type=float,
    default=0,
    help='weight decay',
)
parser.add_argument(
    '--accumulate_grad',
    type=float,
    default=1,
    help='number of steps to accumulate grads',
)
parser.add_argument(
    '--mode',
    type=int,
    default=0,
    help='src (0), trg (1), uda (2)',
)
parser.add_argument(
    '--train',
    action='store_true',
    default=False,
    help='enable training',
)
parser.add_argument(
    '--search',
    action='store_true',
    default=False,
    help='enable hyperparam search',
)
args = parser.parse_args()
