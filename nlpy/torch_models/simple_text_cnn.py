from torch import nn

class SimpleTextCNN(nn.Module):
    def __init__(self, sequence_length=128, num_embeddings=10000,
    embedding_dim=64, padding_idx=1, conv_out_channels=12,
    conv_kernel_size=5, dropout=0.25, output_dim=1):
        super(SimpleTextCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx)
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size)
        self.relu = nn.ReLU()
        self.maxp1d = nn.MaxPool1d(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sequence_length,output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1d(x.permute(0,2,1))
        x = self.relu(x)
        x = self.maxp1d(x)
        x = self.dropout(x.squeeze())
        x = self.linear(x)
        return x