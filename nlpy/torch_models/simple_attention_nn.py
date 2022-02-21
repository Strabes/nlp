import time
import numpy as np
import torch
from collections import Counter, OrderedDict
from torch import nn
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd

def build_vocab(texts, tokenizer, max_tokens=10000,
    oov_token="<OOV>", pad_token = "<PAD>"):
    counter = Counter(
        token for tokens in map(tokenizer, texts)
        for token in tokens)

    prevocab = OrderedDict(counter.most_common(max_tokens))
    for special in [pad_token, oov_token]:
        if not special in prevocab:
            key_to_remove = list(prevocab.keys())[-1]
            del prevocab[key_to_remove]
            prevocab.update({special:0})
            prevocab.move_to_end(special, last=False)

    vocab = torchtext.vocab.vocab(prevocab)
    
    if pad_token not in vocab: vocab.insert_token(pad_token, 0)
    if oov_token not in vocab: vocab.insert_token(oov_token, 0)
    vocab.set_default_index(vocab[oov_token])

    return vocab


class TextDataset(Dataset):
    def __init__(self, text, target, tokenizer, vocab, max_sequence_length=250, padding_token="<PAD>"):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length
        self.padding_token = padding_token

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        target = self.target[idx]
        tokens = self.tokenizer(text)[:self.max_sequence_length]
        tokens = tokens + [self.padding_token]*(self.max_sequence_length - len(tokens))
        seq = [self.vocab[t] for t in tokens]
        return torch.tensor(seq), torch.tensor([target], dtype=torch.float32)


class AttentionNetwork(nn.Module):
    def __init__(self, sequence_length=128, num_embeddings=10000, embedding_dim=64, padding_idx=1,
    att_num_heads=4, dropout=0.25, output_dim=1):
        super(AttentionNetwork, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx)
        self.mhatt = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=att_num_heads,
            batch_first=True)
        self.relu = nn.ReLU()
        self.maxp1d = nn.MaxPool1d(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sequence_length,output_dim)

    def forward(self,x):
        x = self.embedding(x)
        x, _ = self.mhatt(x,x,x)
        x = self.relu(x)
        x = self.maxp1d(x)
        x = self.dropout(x.squeeze())
        x = self.linear(x)
        return x
        

def train_epoch(model, train_dataloader, loss_fn, optimizer, device):

    t0_epoch = time.time()
    model.train()

    epoch_loss = 0
    
    for batch_num, batch in enumerate(train_dataloader):
        batch_features, batch_targets = tuple(t.to(device) for t in batch)
        preds = model(batch_features)
        loss = loss_fn(preds, batch_targets)

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_train_loss = epoch_loss / len(train_dataloader)

    time_elapsed = time.time() - t0_epoch

    return mean_train_loss, time_elapsed

def evaluation(model, dataloader, loss_fn, device):
    t0 = time.time()
    model.eval()

    eval_loss = []

    for batch_num, batch in enumerate(dataloader):
        features, targets = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            preds = model(features)

        # Compute loss
        loss = loss_fn(preds, targets)
        eval_loss.append(loss.item())

    eval_loss = np.mean(eval_loss)

    time_elapsed = time.time() - t0

    return eval_loss, time_elapsed

