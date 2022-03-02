from typing import Iterable, Callable
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

def prep_text(texts:Iterable, tokenizer, max_sequence_length):
    """
    Create generator of padded sequences

    Paramaters
    ----------
    texts : Iterable

    tokenizer : object
        must have `texts_to_sequences` method that transforms
        iterable of strings into iterable of integer sequences
        E.g.: object of type `keras_preprocessing.text.Tokenizer`

    max_sequence_length : int

    Yields
    ------
    numpy.array
        1D numpy.array of length max_sequence_length
    """
    for text in texts:
        text_sequences = tokenizer.texts_to_sequences([text])
        yield sequence.pad_sequences(
            text_sequences, maxlen=max_sequence_length,
            padding='post', truncating='post').reshape(-1)


def dataset_callable(texts, targets, tokenizer, max_sequence_length, batch_size):
    """
    Construct tensorflow dataset

    Parameters
    ----------
    texts : Iterable

    targets : Iterable

    tokenizer : object
        must have `texts_to_sequences` method that transforms
        iterable of strings into iterable of integer sequences
        E.g.: object of type `keras_preprocessing.text.Tokenizer`

    max_sequence_length : int

    batch_size : int

    Returns
    -------
    dataset : Callable
        dataset() is a tensorflow.data.Dataset
    """
    text_prepped = lambda: prep_text(texts, tokenizer, max_sequence_length)
    dataset = lambda: tf.data.Dataset.from_generator(
        lambda: zip(text_prepped(),(i for i in targets)),
        output_signature=(
        tf.TensorSpec(shape=(max_sequence_length),dtype=tf.int32),
        tf.TensorSpec(shape=(),dtype=tf.int32))).batch(batch_size)
    return dataset