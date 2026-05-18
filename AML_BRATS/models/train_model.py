from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.data_loading import BRATSDataset, get_dataset_folds


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
    run_name: str,
    device: torch.device = _get_device(),
) -> tuple[float, float]:
    """
    Train any model with the specified loss function and optimizer.
    Train and validation losses are saved using tensorboard.
    Final loss values are returned, and the final model state is saved.
    """
    writer = SummaryWriter(f"runs/{run_name}")

    train_loss = 0.0
    val_loss = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train_epoch(train_dl, model, loss_fn, optimizer, device)
        val_loss = validation_epoch(validation_dl, model, loss_fn, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Train loss: {train_loss}, validation loss: {val_loss}")

    torch.save(model.state_dict(), f"models/{run_name}_final.pkl")
    return train_loss, val_loss


def train_k_fold(
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    run_name: str,
    batch_size: int = 8,
) -> tuple[float, float]:
    """
    Train a given model for all k folds.
    Returns average train and validation loss across folds.
    """
    folds, _ = get_dataset_folds()
    total_train_loss = 0.0
    total_val_loss = 0.0
    for i, fold in enumerate(folds):
        train_ds = BRATSDataset(fold[0], augmented=True)
        val_ds = BRATSDataset(fold[1])

        train_dl = DataLoader(
            train_ds, batch_size=batch_size, num_workers=4, shuffle=True
        )
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

        print(f"Training fold {i + 1}/{len(folds)}")
        train_loss, val_loss = train_model(
            model,
            train_dl,
            val_dl,
            loss_fn,
            optimizer,
            epochs,
            run_name=f"{run_name}_FOLD{i + 1}",
        )
        total_train_loss += train_loss
        total_val_loss += val_loss

    return total_train_loss / len(folds), total_val_loss / len(folds)
