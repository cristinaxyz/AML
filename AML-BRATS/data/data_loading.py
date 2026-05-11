from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

DATA_PATH = Path("data/BraTS2020_training_data")


def _process_path(path_str: str) -> Path:
    """Convert a single path to work with the project file structure."""
    relative_path = Path(path_str).relative_to("/")
    return DATA_PATH / relative_path


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the metadata and processess all paths."""
    metadata = pd.read_csv(path)
    metadata["slice_path"] = metadata["slice_path"].map(_process_path)
    return metadata


def split_by_volume(
    metadata: pd.DataFrame,
    seed: int = 42,
    train: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the metadata by volume (patient ID)."""
    volumes = metadata["volume"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(volumes)

    n_total = len(volumes)
    n_train = int(n_total * train)

    train_volumes = volumes[:n_train]
    test_volumes = volumes[n_train:]

    train_data = metadata[metadata["volume"].isin(train_volumes)]
    test_data = metadata[metadata["volume"].isin(test_volumes)]

    return train_data, test_data


class BRATSDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame) -> None:
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path = self.metadata.iloc[idx]["slice_path"]

        with h5py.File(path, "r") as f:
            image: np.ndarray = f["image"][()]
            mask: np.ndarray = f["mask"][()]

        return {
            "image": image,
            "mask": mask,
        }

        # Preprocessing and/or data augumentation should go here i think?


metadata = load_metadata(DATA_PATH / "content/data/meta_data.csv")
train_metadata, test_metadata = split_by_volume(metadata)
train_ds = BRATSDataset(train_metadata)
test_ds = BRATSDataset(test_metadata)
print(train_ds[0])
print(test_ds[0])
