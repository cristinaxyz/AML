import torch
from torch.utils.data import DataLoader

from ..data.data_loading import BRATSDataset, get_dataset_folds
from .train_model import train_model
from .unet import UNet

LR = 1e-3
NUM_EPOCHS = 20


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        num = 2 * (probs * targets).sum(dim=(2, 3))
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (num + self.smooth) / (den + self.smooth)
        return 1 - dice.mean()


if __name__ == "__main__":
    folds, _ = get_dataset_folds()
    for i, fold in enumerate(folds):
        model = UNet(3)
        train_ds = BRATSDataset(fold[0], augmented=True)
        val_ds = BRATSDataset(fold[1])

        train_dl = DataLoader(
            train_ds, batch_size=8, num_workers=4, shuffle=True
        )
        val_dl = DataLoader(val_ds, batch_size=8, num_workers=4)

        loss_fn = DiceLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        print(f"Training fold {i + 1}/{len(folds)}")
        train_loss, val_loss = train_model(
            model,
            train_dl,
            val_dl,
            loss_fn,
            optimizer,
            NUM_EPOCHS,
            run_name=f"unet_E{NUM_EPOCHS}_FOLD{i + 1}",
        )
