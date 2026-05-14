from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.optim.SGD


def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    device = device or (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    model.train()

    train_loss = 0.0
    n = 0
    for X, y_true in tqdm(dataloader, desc="Training..."):
        X = X.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)
        train_loss += float(loss)
        n += 1

        loss.backward()
        optimizer.step()

    return train_loss / n if n else 0.0


def validation_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    device: torch.device,
) -> float:
    """Run one validation epoch and return the average loss."""
    model.to(device)
    model.eval()
    val_loss = 0.0
    n = 0

    with torch.no_grad():
        for X, y_true in tqdm(dataloader, desc="Validation..."):
            X = X.to(device)
            y_true = y_true.to(device)

            y_pred = model(X)
            val_loss += float(loss_fn(y_pred, y_true))
            n += 1

    return val_loss / n if n else 0.0


def _get_device() -> torch.device:
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def train_model(
    model: torch.nn.Module,
    train_dl: DataLoader,
    validation_dl: DataLoader,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device = _get_device(),
):
    for i in range(epochs):
        print(f"Epoch {i}")
        train_loss = train_epoch(train_dl, model, loss_fn, optimizer, device)
        val_loss = validation_epoch(validation_dl, model, loss_fn, device)
        print(f"Train loss: {train_loss}, validation loss: {val_loss}")
