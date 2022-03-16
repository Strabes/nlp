import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_enc_ff: int,
                 nlayers: int, out_dim: int, dropout: float = 0.5):
        """
        Parameters
        ----------
        ntoken: int
            number of tokens in the vocabulary
        d_model: int
            dimension of embeddings
        nhead: int
            number of heads
        d_enc_ff: int
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_enc_ff, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, out_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor=None) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)




def make_test_batch(batch_size, seq_len, ntokens):
    import numpy as np
    x = np.random.randint(1,ntokens,(batch_size,seq_len))
    cutoff = np.random.randint(5,seq_len,(batch_size,))
    mask = np.zeros((batch_size, seq_len))
    for i, v in enumerate(cutoff):
        mask[i,:v] = 1
    x = torch.as_tensor(x*mask,dtype=torch.int64)
    return x

if __name__ == '__main__':
    batch_size = 512
    seq_len = 64
    d_model = 16
    ntokens = 1000
    nlayers = 6
    out_dim = 128

    x = make_test_batch(batch_size, seq_len, ntokens)
    tm = TransformerModel(ntoken=ntokens, d_model=d_model,
         nhead=4, d_enc_ff=32, nlayers=nlayers, out_dim=out_dim,
         dropout= 0.5)

    assert tm(x).shape == torch.Size([batch_size,seq_len,out_dim])