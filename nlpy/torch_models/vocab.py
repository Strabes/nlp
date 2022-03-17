import random
import itertools
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#TOKENIZER = get_tokenizer('basic_english')

SPECIAL_TOKENS = ['<UNK>','<PAD>','<CLS>','<SEP>','<EOS>','<MASK>']

def build_vocab(data, tokenizer, default_token = '<UNK>',
    special_tokens = SPECIAL_TOKENS):
    vocab = build_vocab_from_iterator(
        map(tokenizer, data),
        specials=special_tokens)
    vocab.set_default_index(default_token)
    return vocab

class MLMBatch:
    def __init__(self, iter, vocab, batch_size, tokenizer):
        self.iter = iter
        self.vocab = vocab
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def next_batch(self):
        raw_batch = itertools.islice(self.iter,self.batch_size)

    def mask_seq(self, seq):
        output_label = []

        for i, idx in enumerate(seq):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    seq[i] = self.vocab["<MASK>"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    seq[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    pass

                output_label.append(idx)

        return seq, output_label