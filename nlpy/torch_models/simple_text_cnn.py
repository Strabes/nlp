from torch import nn
from math import floor

class SimpleTextCNN(nn.Module):
    """A simple torch CNN for text data"""
    def __init__(self, sequence_length:int=128, num_embeddings:int=10000,
    embedding_dim:int=64, padding_idx:int=1, conv_out_channels:int=12,
    conv_kernel_size:int=5, conv_padding='valid', conv_dilation=1,
    conv_stride=2, dropout:float=0.25, output_dim:int=1):
        """
        Parameters
        ----------
        sequence_length : int
        num_embeddings : int
        embedding_dim : int
        padding_idx : int
        conv_out_channels : int
        conv_kernel_size : int
        dropout : float
        output_dim : int
        """
        super(SimpleTextCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx)
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
            dilation=conv_dilation,
            stride=conv_stride)
        if conv_padding == 'valid':
            self.conv_output_layer_size = floor(
                (sequence_length - conv_dilation*(conv_kernel_size - 1) - 1) /
                conv_stride + 1)
        elif conv_padding == 'same':
            self.conv_output_layer_size = floor(
                (sequence_length - (conv_dilation-1)*(conv_kernel_size - 1) - 1) /
                conv_stride + 1)
        else:
            raise(ValueError(f"conv_padding must be one of 'valid' or 'same'" +
            f"but received {conv_padding}"))
        self.relu = nn.ReLU()
        self.maxp1d = nn.MaxPool1d(self.conv_output_layer_size)
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