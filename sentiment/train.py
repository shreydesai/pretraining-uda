import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AdamW
from sklearn.metrics import classification_report

from main import args
from data import TextDataset, create_loader
from models import BertMLP, RobertaMLP, XLNetMLP
from utils import cuda


def _optimizer_step(args, i):
    return i % args.accumulate_grad == 0


def train(train_ds):
    model.train()
    train_loss = 0.
    global_steps = 0.
    train_loader = create_loader(args, train_ds)
    optimizer.zero_grad()
    for i, (x, y) in enumerate(train_loader, 1):
        loss = criterion(model(x), y)
        if args.accumulate_grad > 1:
            loss = loss / args.accumulate_grad
        train_loss += loss.item()
        loss.backward()
        if _optimizer_step(args, i):
            optimizer.step()
            optimizer.zero_grad()
            global_steps += 1
    return train_loss / global_steps


def valid(valid_ds):
    model.eval()
    valid_loss = 0.
    valid_loader = create_loader(args, valid_ds)
    for (x, y) in valid_loader:
        with torch.no_grad():
            loss = criterion(model(x), y)
        valid_loss += loss.item()
    return valid_loss / len(valid_loader)


def test(test_ds):
    model.eval()
    test_loader = create_loader(args, test_ds, shuffle=False)
    y_true, y_pred = [], []
    for (x, y) in test_loader:
        with torch.no_grad():
            preds = model(x).argmax(1)
        for i in range(x.size(0)):
            y_true.append(y[i].item())
            y_pred.append(preds[i].item())
    return classification_report(y_true, y_pred, digits=6)


if args.model == 'bert-base-uncased':
    model = BertMLP()
elif args.model == 'roberta-base':
    model = RobertaMLP()
elif args.model == 'xlnet-base-cased':
    model = XLNetMLP()
else:
    raise NotImplementedError


model = cuda(args, model)
if args.pretrained_ckpt:
    pretrained_state_dict = torch.load(
        args.pretrained_ckpt, map_location='cpu'
    )
    for n, p in model.named_parameters():
        if n in pretrained_state_dict:
            w = pretrained_state_dict[n]
            p.data.copy_(w.data)
    model = cuda(args, model)
    print('loaded pretrained ckpt')

optimizer = AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)
criterion = nn.CrossEntropyLoss()
best_loss = float('inf')

train_ds = TextDataset(f'amazon/{args.src}_train.csv', args.src_p)
valid_ds = TextDataset(f'amazon/{args.src}_valid.csv', args.src_p)
test_ds = TextDataset(f'amazon/{args.src}_test.csv', args.src_p)

if args.train:
    for epoch in range(1, args.epochs + 1):
        train_loss = train(train_ds)
        valid_loss = valid(valid_ds)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), args.ckpt)
        print(
            f'epoch: {epoch} | '
            f'train loss: {train_loss:.6f} | '
            f'valid loss: {valid_loss:.6f}'
        )

model.load_state_dict(torch.load(args.ckpt))
clf_report = test(test_ds)
print(clf_report)
