from torch import nn


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
        self.linear = nn.Linear(embedding_dim,embedding_dim)
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
