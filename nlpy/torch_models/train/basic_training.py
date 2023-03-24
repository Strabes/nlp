import time
import torch
import numpy as np

def train_epoch(model, train_dataloader, loss_fn, optimizer, device, scheduler=None):
    """
    Training a basic torch text model

    Parameters
    ----------
    model: torch.nn.Module

    train_dataloader: torch.utils.data.DataLoader
    """
    t0_epoch = time.time()
    model.train()

    epoch_loss = 0
    
    for batch_num, batch in enumerate(train_dataloader):
        batch_features, batch_targets = tuple(t.to(device) for t in batch)
        preds = model(batch_features)
        loss = loss_fn(preds, batch_targets)

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    mean_train_loss = epoch_loss / len(train_dataloader)

    time_elapsed = time.time() - t0_epoch

    return mean_train_loss, time_elapsed

def evaluation(model, dataloader, loss_fn, device):
    t0 = time.time()
    model.eval()

    eval_loss = []

    for batch_num, batch in enumerate(dataloader):
        features, targets = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            preds = model(features)

        # Compute loss
        loss = loss_fn(preds, targets)
        eval_loss.append(loss.item())

    eval_loss = np.mean(eval_loss)

    time_elapsed = time.time() - t0

    return eval_loss, time_elapsed