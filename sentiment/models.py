import torch.nn as nn
from transformers import BertModel, RobertaModel

from main import args


class BertMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 5)

    def forward(self, x):
        reps, _ = self.bert(x)  # [bs, seq_len, 768]
        cls_rep = reps[:, 0]  # [bs, 768]
        logits = self.fc(cls_rep)  # [bs, 5]
        return logits


class RobertaMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Linear(768, 5)

    def forward(self, x):
        reps, _ = self.roberta(x)  # [bs, seq_len, 768]
        cls_rep = reps[:, 0]  # [bs, 768]
        logits = self.fc(cls_rep)  # [bs, 5]
        return logits
