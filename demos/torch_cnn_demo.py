# %%
import time
import numpy as np
from nlpy.torch_models.simple_attention_nn import AttentionNetwork
from nlpy.torch_models.preprocess import (
    build_vocab,
    TextDataset,
    train_epoch,
    evaluation)
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("../data/yelp.csv")
train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(test_val_df,test_size=0.5, random_state=42)

# %%
hparams = {
    "batch_size": 128,
    "conv_channels": 24,
    "conv_kernel_size": 4,
    "conv_stride": 2,
    "conv_padding": 'valid',
    "conv_dilation": 1,
    "embedding_dim": 24,
    "dropout_rate": 0.25,
    "output_size": 1,
    "learning_rate": 0.005,
    "max_num_words": 2000,
    "max_sequence_length": 200}

# %%
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab(
    train_df["text"],
    tokenizer,
    max_tokens=hparams["max_num_words"],
    oov_token="<OOV>",
    pad_token = "<PAD>")

# %%
train_dataset = TextDataset(
    train_df["text"].tolist(),
    train_df["stars"].tolist(),
    max_sequence_length=hparams["max_sequence_length"],
    tokenizer=tokenizer,
    vocab=vocab)

val_dataset = TextDataset(
    val_df["text"].tolist(),
    val_df["stars"].tolist(),
    max_sequence_length=hparams["max_sequence_length"],
    tokenizer=tokenizer,
    vocab=vocab)

# %%
train_dataloader = DataLoader(train_dataset,hparams["batch_size"],shuffle=True)
val_dataloader = DataLoader(val_dataset,hparams["batch_size"],shuffle=True)

# %%
model = AttentionNetwork(
    sequence_length=hparams["max_sequence_length"],
    num_embeddings=len(vocab),
    embedding_dim=hparams["embedding_dim"],
    padding_idx=1,
    att_num_heads=4,
    dropout=0.25,
    output_dim=1)

# %%
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])

# %%
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# %%
model.to(device)
for epoch in range(5):
    mean_train_loss, time_elapsed = train_epoch(
        model, train_dataloader, loss_fn, optimizer, device)
    print(f"Training Epoch: {epoch}")
    print(f"Training mean loss: {mean_train_loss:^8.3f}")
    print(f"Training time elapsed: {time_elapsed:^8.2f}")
    eval_loss, time_elapsed = evaluation(model, val_dataloader, loss_fn, device) 
    print(f"Val mean loss: {eval_loss:^8.3f}")
    print(f"Val time elapsed: {time_elapsed:^8.2f}")

# %%
torch.__version__


