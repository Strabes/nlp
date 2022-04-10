import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq, pad_idx=0):
    seq = tf.cast(tf.math.equal(seq, pad_idx), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, key_dim, value_dim, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            value_dim=value_dim, dropout=dropout_rate)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):

        attn_output = self.mha(x, x, x, mask, return_attention_scores=False)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
    num_layers:int=6,
    d_model:int=512,
    key_dim:int=512,
    value_dim:int=512,
    num_heads:int=8,
    dff:int=2048,
    input_vocab_size:int=50000,
    maximum_position_encoding:int=512,
    dropout_rate:float=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, key_dim, value_dim, dff, dropout_rate)
                       for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


def make_test_batch(batch_size, seq_len, ntokens):
    x = np.random.randint(1,ntokens,(batch_size,seq_len))
    cutoff = np.random.randint(5,seq_len,(batch_size,))
    mask = np.zeros((batch_size, seq_len))
    for i, v in enumerate(cutoff):
        mask[i,:v] = 1
    x = tf.cast(x*mask,dtype=tf.int16)
    return x

if __name__ == '__main__':
    batch_size = 512
    seq_len = 64
    d_model = 16
    ntokens = 1000
    num_layers=2
    num_heads=4
    dff=32
    dropout_rate=0.1
    key_dim = value_dim = 8

    x = make_test_batch(batch_size, seq_len, ntokens)

    encoder = Encoder(
    num_layers=num_layers,
    d_model=d_model,
    key_dim=key_dim,
    value_dim=value_dim,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=ntokens,
    maximum_position_encoding=seq_len,
    dropout_rate=dropout_rate)

    out_shape = encoder(x,training=False,mask=None).shape
    shape_intended = tf.TensorShape([batch_size,seq_len,d_model])
    assert out_shape == shape_intended
