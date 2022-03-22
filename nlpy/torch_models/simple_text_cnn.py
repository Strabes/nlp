import torch
import torch.nn.functional as F
from torch import nn
from math import floor
import numpy as np


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
            token sequence length
        num_embeddings : int
            number of tokens in vocab
        embedding_dim : int
            token embedding dimension
        padding_idx : int
            padding token index
        conv_out_channels : int
            number of convolution channels
        conv_kernel_size : int
            convolution kernel size
        dropout : float
            dropout rate
        output_dim : int
            output dimension
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
        # self.relu = nn.ReLU()
        self.maxp1d = nn.MaxPool1d(self.conv_output_layer_size)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sequence_length,output_dim)

    def forward(self, x):
        x = self.embedding(x) # (batch_size, sequence_length, embedding_dim)
        x = self.conv1d(x.permute(0,2,1)) # (batch_size, conv_out_channels, conv_output_layer_size)
        # x = self.relu(x)
        x = self.maxp1d(x)
        x = self.dropout(x.squeeze())
        x = self.linear(x)
        return x




class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 output_dim=2,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            output_dim (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits