# %%
import inspect
from nlpy.torch_models.simple_text_cnn import CNN_NLP
from nlpy.torch_models.preprocess import (
    build_vocab,
    TextDataset)
from nlpy.torch_models.train.basic_training import (
    train_epoch,
    evaluation)
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from nlpy.utils import load_config

# %%
def main(config, data_loc):
    df = pd.read_csv(data_loc)
    train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(test_val_df,test_size=0.5, random_state=42)

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(
        train_df["text"],
        tokenizer,
        max_tokens=config["vocab_size"],
        oov_token="<OOV>",
        pad_token = "<PAD>")

    train_dataset = TextDataset(
        train_df["text"].tolist(),
        train_df["stars"].tolist(),
        max_sequence_length=config["max_sequence_length"],
        tokenizer=tokenizer,
        vocab=vocab)

    val_dataset = TextDataset(
        val_df["text"].tolist(),
        val_df["stars"].tolist(),
        max_sequence_length=config["max_sequence_length"],
        tokenizer=tokenizer,
        vocab=vocab)

    train_dataloader = DataLoader(train_dataset,config["batch_size"],shuffle=True)
    val_dataloader = DataLoader(val_dataset,config["batch_size"],shuffle=True)

    cnn_params = {k: v for k, v in config.items()
        if k in [p.name for p in inspect.signature(CNN_NLP.__init__).parameters.values()]}
    model = CNN_NLP(**cnn_params)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

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


if __name__ == '__main__':
    from pathlib import Path
    config_loc = Path(__file__).parent / 'configs/torch_cnn.json'
    config = load_config(config_loc)
    data_loc = Path(__file__).parent.parent / config["data_path"]
    main(config,data_loc)