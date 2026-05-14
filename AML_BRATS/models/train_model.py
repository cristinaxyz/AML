from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.to(device)
    model.train()

    train_loss = 0.0
    total_samples = 0

    for datapoint in tqdm(dataloader, desc="Training"):
        X = datapoint["image"]
        y_true = datapoint["mask"]
        X = X.to(device)
        y_true = y_true.to(device)

        batch_size = X.size(0)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)

        train_loss += loss.item() * batch_size
        total_samples += batch_size

        loss.backward()
        optimizer.step()

    return train_loss / total_samples if total_samples else 0.0


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
    total_samples = 0

    with torch.no_grad():
        for datapoint in tqdm(dataloader, desc="Validation"):
            X = datapoint["image"]
            y_true = datapoint["mask"]
            X = X.to(device)
            y_true = y_true.to(device)

            batch_size = X.size(0)
            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)

            val_loss += loss.item() * batch_size
            total_samples += batch_size

    return val_loss / total_samples if total_samples else 0.0


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
) -> tuple[float, float]:
    train_loss = 0.0
    val_loss = 0.0

    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_loss = train_epoch(train_dl, model, loss_fn, optimizer, device)
        val_loss = validation_epoch(validation_dl, model, loss_fn, device)
        print(f"Train loss: {train_loss}, validation loss: {val_loss}")

    return train_loss, val_loss
