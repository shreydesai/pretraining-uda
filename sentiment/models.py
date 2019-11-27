import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, XLNetModel

from main import args
from data import tokenizer
from utils import cuda


class BertMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 5)

    def forward(self, x, x_len=None):
        reps, _ = self.bert(
            x,
            attention_mask=(x != tokenizer.pad_token_id).float(),
        )  # [bs, seq_len, 768]
        cls_rep = reps[:, 0]  # [bs, 768]
        logits = self.fc(cls_rep)  # [bs, 5]
        return logits


class RobertaMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Linear(768, 5)

    def forward(self, x, x_len=None):
        reps, _ = self.roberta(
            x,
            attention_mask=(x != tokenizer.pad_token_id).float(),
        )  # [bs, seq_len, 768]
        cls_rep = reps[:, 0]  # [bs, 768]
        logits = self.fc(cls_rep)  # [bs, 5]
        return logits


class XLNetMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.fc = nn.Linear(768, 5)

    def forward(self, x, x_len):
        reps = self.xlnet(
            x,
            attention_mask=(x != tokenizer.pad_token_id).float(),
        )  # [bs, seq_len, 768]
        cls_rep = reps[0][
            cuda(torch.arange(x.size(0))),
            x_len - 1,
        ]  # [bs, 768]
        logits = self.fc(cls_rep)  # [bs, 5]
        return logits
