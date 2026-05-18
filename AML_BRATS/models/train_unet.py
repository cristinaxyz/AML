import torch

from .train_model import train_k_fold
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
    model = UNet(3)
    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_k_fold(
        model,
        loss_fn,
        optimizer,
        epochs=NUM_EPOCHS,
        run_name=f"UNET_20EPOCHS_{LR}LR",
    )
