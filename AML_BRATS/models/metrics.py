import torch


def calculate_dice(
    probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5
) -> torch.Tensor:
    """Calculate Dice score for probability or binary masks."""
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (num + smooth) / (den + smooth)
    valid_channels = targets.sum(dim=(2, 3)) > 0

    if valid_channels.any():
        return dice.masked_select(valid_channels).mean()

    return dice.new_tensor(0.0)