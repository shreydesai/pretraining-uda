import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AdamW
from sklearn.metrics import classification_report

from main import args
from data import get_datasets, create_loader
from models import BertMLP
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


model = cuda(args, BertMLP())
optimizer = AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)
criterion = nn.CrossEntropyLoss()
best_loss = float('inf')

src_train, src_valid, src_test = get_datasets(
    args.src, args.search
)
trg_train, trg_valid, trg_test = get_datasets(
    args.trg, args.search
)

if args.mode == 0 or args.mode == 2:
    train_ds = src_train
    valid_ds = src_valid
    test_ds = src_test
elif args.mode == 1:
    train_ds = trg_train
    valid_ds = trg_valid
    test_ds = trg_test
else:
    raise RuntimeError

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
clf_report = test(trg_test)
print(clf_report)
