import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dropout, Embedding


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



def make_test_batch(batch_size, seq_len, ntokens):
    x = np.random.randint(1,ntokens,(batch_size,seq_len))
    cutoff = np.random.randint(5,seq_len,(batch_size,))
    mask = np.zeros((batch_size, seq_len))
    for i, v in enumerate(cutoff):
        mask[i,:v] = 1
    x = tf.cast(x*mask,dtype=tf.int16)
    return x

batch_size = 512
seq_len = 64
d_model = 16
ntokens = 1000

x = make_test_batch(batch_size, seq_len, ntokens)

emb = Embedding(input_dim=ntokens,output_dim=d_model)

embedded = emb(x)

pos_enc = positional_encoding(position=seq_len, d_model=d_model)

final_emb = embedded + pos_enc



layer = MultiHeadAttention(num_heads=4, key_dim=8)
incoming = tf.keras.Input(shape=[8, 16])
output_tensor, weights = layer(incoming, incoming,
                               return_attention_scores=True)
print(output_tensor.shape)

print(weights.shape)