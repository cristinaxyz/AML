import io

import h5py
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from AML_BRATS.data.data_loading import preprocess
from AML_BRATS.models.unet import UNet

MODEL_PATH = "models/UNET_HYD_25EPOCHS_adam_BNORM_LR0.0001_WD0.01_bce1_NOAUG_BS64_FOLD1_final.pkl"
MODEL_BATCH_NORM = "BNORM" in MODEL_PATH


class HealthResponse(BaseModel):
    message: str


def load_h5_input(file: UploadFile = File(...)) -> np.ndarray:
    """
    Extracts data from a h5 file.

    If the file is not h5, raise error.
    """
    try:
        with h5py.File(file.file, "r") as f:
            key = list(f.keys())[0]
            if len(key) == 0:
                raise HTTPException(
                    status_code=400, detail="No scan found in this file"
                )
            input_data = np.array(f[key])
            return input_data
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid '.h5' file.")
    except Exception as e:
        raise HTTPException(
            status_code=415, detail=f"Invalid '.h5' file: {str(e)}"
        )


def segmentation_to_png(prediction: torch.Tensor, threshold: float = 0.5):
    prediction_arr: np.ndarray = prediction.detach().cpu().numpy()
    binary_mask = prediction_arr >= threshold
    rgb = np.zeros(
        (prediction_arr.shape[1], prediction_arr.shape[2], 3), dtype=np.uint8
    )
    rgb[binary_mask[0]] = [255, 0, 0]
    rgb[binary_mask[1]] = [0, 255, 0]
    rgb[binary_mask[2]] = [0, 0, 255]

    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


app = FastAPI(
    title="Brain Tumor Segmentation",
    description="""API for Brain Tumor Segmentation

    Our model performs segmentation on MRI scan slices. We trained (model info).

    It was trained on BraTS2020 dataset, which included MRI scan slices with segmentations performed manually by neuro-radiologists.

    User guide:
    1. Scroll to "Predict" section. 
    2. Click on "Try it out" button.
    3. Click on "Browse..." and select the scan file to be uploaded from your computer.
    4. Click on the blue button "Execute".
    5. Scroll to Response Body: you may visualize or save the '.png' file with the regions of the tumor from the uploaded scan.
    
    Important!
    This API is for educational and testing purposes only.
    This is not a diagnostic tool!

    Input:
    Upload a '.h5' file with a slice from a MRI scan.

    Output:
    Segmentation mask as a '.png' file:
    - red, green, blue = regions of the tumor:
        * the necrotic and non-enhancing tumor core 
        * the peritumoral edema
        * the GD-enhancing tumor
    - black = background/healthy tissue
    """,
    version="alpha",
)


@app.get("/", description="Root endpoint")
async def root():
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    description="Check if the API is running.",
)
async def health():
    return {"message": "API is running"}


@app.post(
    "/predict",
    description="Upload a '.h5' MRI scan slice. Returns a '.png' file with the segmentation mask with 3 regions of the tumor identified.",
)
async def predict(scan: UploadFile):
    if scan.filename is None:
        raise HTTPException(
            status_code=400, detail="Invalid or no file uploaded"
        )
    if not scan.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="File must be '.h5'")
    input_data = load_h5_input(scan)
    input_data = preprocess(input_data)
    input_tensor = torch.from_numpy(input_data).unsqueeze(0)
    input_tensor = input_tensor.permute(0, 3, 1, 2).float()

    with torch.no_grad():
        prediction = model(input_tensor).sigmoid()[0]

    user_image = segmentation_to_png(prediction)
    return StreamingResponse(user_image, media_type="image/png")


def main():
    global model
    model = UNet(3, MODEL_BATCH_NORM)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
