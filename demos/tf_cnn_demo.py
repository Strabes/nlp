# %%
import time
import pandas as pd
import tensorflow as tf
import json
import inspect
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from nlpy.tf_models.preprocess import dataset_callable
from nlpy.tf_models.simple_text_cnn import SimpleTextCNN
from nlpy.tf_models.train.basic_training import train

def load_config(config):
    with open(config,"r") as f:
        config = json.load(f)
    return config


# %%
def main(config, data_loc):
    df = pd.read_csv(data_loc)
    train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(test_val_df,test_size=0.5, random_state=42)
    tokenizer = Tokenizer(num_words=config["max_sequence_length"])
    tokenizer.fit_on_texts(train_df["text"])

    train_dataset = dataset_callable(
    train_df["text"], train_df["stars"], tokenizer,
    config["max_sequence_length"], config["batch_size"])

    test_dataset = dataset_callable(
    test_df["text"], test_df["stars"], tokenizer,
    config["max_sequence_length"], config["batch_size"])

    val_dataset = dataset_callable(
    val_df["text"], val_df["stars"], tokenizer,
    config["max_sequence_length"], config["batch_size"])

    tf.random.set_seed(42)
    cnn_params = {k: v for k, v in config.items()
        if k in [p.name for p in inspect.signature(SimpleTextCNN.__init__).parameters.values()]}
    model = SimpleTextCNN(**cnn_params)

    loss_object = tf.keras.losses.MeanSquaredError()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_rmse = tf.keras.metrics.RootMeanSquaredError(name='train_rmse')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_rmse = tf.keras.metrics.RootMeanSquaredError(name='val_rmse')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_rmse = tf.keras.metrics.RootMeanSquaredError(name='test_rmse')

    train(model, optimizer, loss_object, train_loss, train_dataset, train_rmse,
    val_loss, val_dataset, val_rmse, epochs=config["epochs"])


if __name__ == '__main__':
    from pathlib import Path
    config_loc = Path(__file__).parent / 'configs/tf_simple_cnn.json'
    config = load_config(config_loc)
    data_loc = Path(__file__).parent.parent / config["data_path"]
    main(config,data_loc)