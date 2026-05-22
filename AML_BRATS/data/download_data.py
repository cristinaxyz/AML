import kagglehub
import shutil
from pathlib import Path

# Download dataset from Kaggle
download_path = Path(
    kagglehub.dataset_download("awsaf49/brats2020-training-data")
)

print("Dataset downloaded to:")
print(download_path)

# Project data folder
project_data = Path("data")
project_data.mkdir(exist_ok=True)

# Copy everything from KaggleHub cache into your project's data folder
for item in download_path.iterdir():
    destination = project_data / item.name

    if item.is_dir():
        if destination.exists():
            print(f"Skipping existing folder: {destination}")
        else:
            shutil.copytree(item, destination)
            print(f"Copied folder: {destination}")
    else:
        if destination.exists():
            print(f"Skipping existing file: {destination}")
        else:
            shutil.copy2(item, destination)
            print(f"Copied file: {destination}")

print("Done.")