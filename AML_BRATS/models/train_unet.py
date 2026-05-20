import torch

from .train_model import train_k_fold
from .unet import UNet

LR = 6e-2
NUM_EPOCHS = 20

def calculate_dice(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (num + smooth) / (den + smooth)
    return dice.mean()

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return 1 - calculate_dice(logits.sigmoid(), targets, self.smooth)


if __name__ == "__main__":
    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    loss_fn = lambda logits, targets: bce_loss(logits, targets) + dice_loss(logits, targets)

    def model_fn():
        return UNet(3)

    def optimizer_fn(params):
        return torch.optim.SGD(params, lr=LR, momentum=0.9)

    train_k_fold(
        model_fn,
        optimizer_fn,
        loss_fn,
        metrics={"dice": calculate_dice},
        epochs=NUM_EPOCHS,
        run_name=f"UNET_DICEBCE_SGD_MOM0.9_{NUM_EPOCHS}EPOCHS_{LR}LR",
    )
