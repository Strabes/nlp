"""
A simple tensorflow text CNN
"""

from math import floor
import pandas as pd
from typing import Iterable, Callable
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Conv1D, ReLU,
    MaxPool1D, Dense, Dropout)
from tensorflow.keras import Model


class SimpleTextCNN(Model):
    """
    Simple tensorflow text CNN
    """
    def __init__(self,sequence_length=64,embedding_input_dim=1024,
        embedding_dim=64, conv_channels=24,conv_kernel_size=4,conv_stride=2,
        conv_padding='valid',conv_dilation=1, dropout_rate=0.25,
        output_size=1):
        """
        Parameters
        ----------
        sequence_length : int
            input sequence length
        embedding_input_dim : int
            number of tokens in vocabulary
        embedding_dim : int
            embedding dimension
        conv_channels : int
            number of convolution output channels
        conv_kernel_size : int
            convolution kernel size
        conv_stride : int
            convolution stride
        conv_padding : str
            convolution padding, either 'valid' or 'same'
        conv_dilation : int
            convolution dilation, >=1
        dropout_rate : float
            dropout rate, [0,1)
        output_size : int
            output size
        """
        super(SimpleTextCNN, self).__init__()
        self.embedding = Embedding(
            input_dim=embedding_input_dim,
            output_dim=embedding_dim,
            input_length=sequence_length)
        self.conv1d = Conv1D(
            filters=conv_channels,
            kernel_size=conv_kernel_size,
            strides=conv_stride,
            padding=conv_padding,
            data_format='channels_last',
            dilation_rate=conv_dilation)
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
        self.relu = ReLU()
        self.maxp1d = MaxPool1D(pool_size=self.conv_output_layer_size)
        self.dropout = Dropout(rate=dropout_rate)
        self.dense = Dense(units=output_size)

    def call(self,x):
        x = self.embedding(x)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxp1d(x)
        x = self.dropout(x)
        x = tf.squeeze(x)
        x = self.dense(x)
        return x


@tf.function
def train_step(model, features, targets, loss_object, optimizer,
    train_loss, train_rmse):
    """
    Training step

    Parameters
    ----------
    model : 
    """
    with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(features, training=True)
        loss = loss_object(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_rmse(targets, predictions)

@tf.function
def test_step(model, features, targets, loss_object, test_loss, test_rmse):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    predictions = model(features, training=False)
    t_loss = loss_object(targets, predictions)

    test_loss(t_loss)
    test_rmse(targets, predictions)