import time
import numpy as np
import torch
from collections import Counter, OrderedDict
import torchtext
from torch.utils.data import Dataset

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
