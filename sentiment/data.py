import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    GPT2Tokenizer,
)

from main import args
from utils import cuda, rpad


if args.model == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.model == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
elif args.model == 'gpt2':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
else:
    raise NotImplementedError


def prepare_text(text):
    tokens = tokenizer.tokenize(text)[:args.seq_len - 2]
    token_ids = tokenizer.encode(tokens, add_special_tokens=True)
    rspace = args.seq_len - len(token_ids)  # <= args.seq_len
    token_ids = rpad(token_ids, rspace, tokenizer.pad_token_id)
    return cuda(args, torch.tensor(token_ids)).long()


def prepare_label(label):
    return cuda(args, torch.tensor(label)).long()


class TextDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)  # args.src or args.trg
        self.seq_len = args.seq_len
        self._cache = {}

    def unpack(self, i):
        entry = self.df.iloc[i]
        return (str(entry['text']), entry['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        res = self._cache.get(i, None)
        if res is None:
            text, label = self.unpack(i)
            text_tensor = prepare_text(text)
            label_tensor = prepare_label(label)
            res = (text_tensor, label_tensor)
            self._cache[i] = res  # cache inputs
        return res


def get_datasets(ds, search=False):
    train = TextDataset(f'amazon/{ds}_train.csv')
    valid = TextDataset(f'amazon/{ds}_valid.csv')
    test = valid if search else TextDataset(f'amazon/{ds}_test.csv')
    return (train, valid, test)


def create_loader(args, ds, shuffle=True):
    return DataLoader(ds, args.bs, shuffle)


def create_loader_multiple(args, ds_list, shuffle=True):
    return DataLoader(ConcatDataset(ds_list), args.bs, shuffle)
