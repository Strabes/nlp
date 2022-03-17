# %%
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from nlpy.tf_models.preprocess import prep_text, dataset_callable
from nlpy.tf_models.simple_text_cnn import SimpleTextCNN, train_step, test_step
tf.__version__

# %%
tf.config.list_physical_devices()

# %%
df = pd.read_csv("../data/yelp.csv")
train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(test_val_df,test_size=0.5, random_state=42)

# %%
df.head()

# %%
hparams = {
    "batch_size": 128,
    "conv_channels": 24,
    "conv_kernel_size": 4,
    "conv_stride": 2,
    "conv_padding": 'valid',
    "conv_dilation": 1,
    "embedding_dim": 100,
    "dropout_rate": 0.25,
    "output_size": 1,
    "learning_rate": 0.005,
    "max_num_words": 10000,
    "max_sequence_length": 250}

# %%
tokenizer = Tokenizer(num_words=hparams["max_num_words"])
tokenizer.fit_on_texts(train_df["text"])

# %%
train_dataset = dataset_callable(
    train_df["text"], train_df["stars"], tokenizer,
    hparams["max_sequence_length"], hparams["batch_size"])

test_dataset = dataset_callable(
    test_df["text"], test_df["stars"], tokenizer,
    hparams["max_sequence_length"], hparams["batch_size"])

val_dataset = dataset_callable(
    val_df["text"], val_df["stars"], tokenizer,
    hparams["max_sequence_length"], hparams["batch_size"])

# %%
tf.random.set_seed(42)
model = SimpleTextCNN(
  sequence_length=hparams["max_sequence_length"],
  embedding_input_dim=hparams["max_num_words"],
  embedding_dim=hparams["embedding_dim"],
  conv_channels=hparams["conv_channels"],
  conv_kernel_size=hparams["conv_kernel_size"],
  conv_stride=hparams["conv_stride"],
  conv_padding=hparams["conv_padding"],
  conv_dilation=hparams["conv_dilation"],
  dropout_rate=hparams["dropout_rate"],
  output_size=hparams["output_size"])

loss_object = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_rmse = tf.keras.metrics.RootMeanSquaredError(name='train_rmse')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_rmse = tf.keras.metrics.RootMeanSquaredError(name='val_rmse')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_rmse = tf.keras.metrics.RootMeanSquaredError(name='test_rmse')

# %%
def train(model, train_loss, train_dataset, train_rmse,
    val_loss, val_dataset, val_rmse, epochs=5):
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train RMSE':^9} " +
          f"| {'Val Loss':^10} | {'Val RMSE':^9} | {'Elapsed':^9}")
    print("-"*60)
    best_val_loss = None
    for epoch in range(epochs):
        t0_epoch = time.time()
        train_loss.reset_state()
        train_rmse.reset_state()
        val_loss.reset_state()
        val_rmse.reset_state()

        for train_features, train_targets in train_dataset():
            train_step(model, train_features, train_targets,
            loss_object, optimizer, train_loss, train_rmse)

        for val_features, val_targets in val_dataset():
            test_step(model, val_features, val_targets,
            loss_object, val_loss, val_rmse)

        time_elapsed = time.time() - t0_epoch
        print(f"{epoch+1:^7} | {train_loss.result():^12.6f} | {train_rmse.result():^9.2f}" +
              f" | {val_loss.result():^10.6f}" +
              f" | {val_rmse.result():^9.2f} | {time_elapsed:^9.2f}")

        if best_val_loss is not None and val_loss.result() > best_val_loss:
            print("Stopping early: Val loss increased")
            break
        else:
            best_val_loss = val_loss.result()
    print(f"Training completed! Final validation RMSE: {val_rmse.result():.2f}.")



# %%
train(model, train_loss, train_dataset, train_rmse, val_loss, val_dataset, val_rmse, epochs=25)

# %%



