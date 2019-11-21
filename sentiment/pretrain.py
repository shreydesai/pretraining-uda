import argparse
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW, BertForMaskedLM

from main import args
from data import tokenizer, get_datasets, create_loader_multiple
from utils import cuda, optimizer_params


def _optimizer_step(args, i):
    return i % args.accumulate_grad == 0


def _create_pretraining_inputs_bert(inputs):
    """Adapted from HuggingFace Transformers (http://bit.ly/2NUxZtM)"""

    def _create_bernoulli_mask(inputs, p):
        return torch.bernoulli(torch.full(inputs.size(), p)).bool()

    labels = inputs.clone()

    # sample tokens for masked LM
    prob_matrix = torch.full(labels.size(), 0.15)
    cls_mask = (inputs == tokenizer.cls_token_id)
    sep_mask = (inputs == tokenizer.sep_token_id)
    pad_mask = (inputs == tokenizer.pad_token_id)
    prob_matrix[(cls_mask | sep_mask | pad_mask)] = 0.0
    masked_idxs = torch.bernoulli(prob_matrix).bool()
    labels[~masked_idxs] = -1  # compute loss on [MASK]

    # replace 80% with [MASK]
    idxs_replaced = _create_bernoulli_mask(labels, 0.8) & masked_idxs
    inputs[idxs_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token
    )

    # replace 10% with rand
    idxs_random = (
        _create_bernoulli_mask(labels, 0.5) &
        masked_idxs &
        ~idxs_replaced
    )
    random_words = torch.randint(
        len(tokenizer), labels.size(), dtype=torch.long
    ).to(args.device)
    inputs[idxs_random] = random_words[idxs_random]

    # replace 10% with same
    # (do nothing)

    return (inputs, labels)


def create_pretraining_inputs(inputs):
    if args.model == 'bert-base-uncased':
        return _create_pretraining_inputs_bert(inputs)
    raise NotImplementedError


def train(train_ds_list):
    model.train()
    train_loss = 0.
    global_steps = 0.
    train_loader = create_loader_multiple(args, train_ds_list)
    optimizer.zero_grad()
    for i, (x, _) in enumerate(train_loader, 1):
        x_in, x_out = create_pretraining_inputs(x)
        logits = model(x_in)[0].transpose(1, 2)  # [bs, vocab, seq_len]
        loss = criterion(logits, x_out)
        if args.accumulate_grad > 1:
            loss = loss / args.accumulate_grad
        train_loss += loss.item()
        loss.backward()
        if _optimizer_step(args, i):
            optimizer.step()
            optimizer.zero_grad()
            global_steps += 1
    return train_loss / global_steps


def test(test_ds_list):
    model.eval()
    test_loss = 0.
    test_loader = create_loader_multiple(args, test_ds_list)
    for (x, _) in test_loader:
        with torch.no_grad():
            x_in, x_out = create_pretraining_inputs(x)
            logits = model(x_in)[0].transpose(1, 2)  # [bs, vocab, seq_len]
            loss = criterion(logits, x_out)
        test_loss += loss.item()
    return test_loss / len(test_loader)


model = cuda(args, BertForMaskedLM.from_pretrained('bert-base-uncased'))
optimizer = AdamW(
    optimizer_params(model),
    lr=args.lr,
    weight_decay=args.wd,
    eps=args.eps,
)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
best_loss = float('inf')

src_train, src_valid, src_test = get_datasets(
    args.src, args.search
)
trg_train, trg_valid, trg_test = get_datasets(
    args.trg, args.search
)

train_ds_list = [src_train, trg_train]
valid_ds_list = [src_valid, trg_valid]
test_ds_list = [src_valid, trg_valid]

if args.train:
    for epoch in range(1, args.epochs + 1):
        train_loss = train(train_ds_list)
        valid_loss = test(valid_ds_list)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), args.ckpt)
        print(
            f'epoch: {epoch} | '
            f'train loss: {train_loss:.6f} | '
            f'valid loss: {valid_loss:.6f}'
        )

model.load_state_dict(torch.load(args.ckpt))
test_loss = test(test_ds_list)
print(f'test loss: {test_loss:.6f}')
