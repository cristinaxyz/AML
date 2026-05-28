import uvicorn

import numpy as np
import h5py
import io
from PIL import Image

from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse

from pydantic import BaseModel

# import model

class HealthResponse(BaseModel):
    message: str

def create_random_file():
    data = np.random.rand(256, 256)
    with h5py.File("test_api.h5", "w") as f:
        f.create_dataset("scan", data=data)

def fake_prediction():
    mask = np.zeros((256, 256))
    mask[50:100, 50:100] = 1
    mask [20:50, 50:55] = 2
    mask [25:50, 78:85] = 3
    return mask

def load_h5_input(file: UploadFile = File(...)) -> np.ndarray:
    """
    Extracts data from a h5 file.

    If the file is not h5, raise error.
    """
    try:
        with h5py.File(file.file, "r") as f:
            if len(list(f.keys())) == 0:
                raise HTTPException(status_code=400, detail="No scan found in this file")
            key=(list(f.keys()))[0]
            input_data = np.array(f[key])
            return input_data 
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid '.h5' file.")
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Invalid '.h5' file: {str(e)}")

def preprocess(data: np.ndarray) -> np.ndarray:
    """
    Normalization of the channel values from the user's MRI input.
    """
    image = data.astype(np.float32)
    brain_mask = np.any(image > 0, axis=-1)
    for channel in range(data.shape[-1]):
        channel_data = image[:, :, channel]
        brain_pixels = channel_data[brain_mask]
        if len(brain_pixels) > 0:
            mean = brain_pixels.mean()
            std = brain_pixels.std()
            if std > 0:
                channel_data[brain_mask] = (channel_data[brain_mask] - mean) / std
        channel_data[~brain_mask] = 0
        image[:, :, channel] = channel_data
    return image

def segmentation_to_png(prediction: np.ndarray):
    """
    Each label corresponds to one color (red, green, blue).
    Converting them into a '.png' from the mask version np.ndarray.
    """
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[prediction == 1] = [255, 0, 0]
    rgb[prediction == 2] = [0, 255, 0]
    rgb[prediction == 3] = [0, 0, 255]
    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG")
    buffer.seek(0)
    return buffer

app = FastAPI(
    title="Brain Tumor Segmentation",
    description = """API for Brain Tumor Segmentation



    Our model performs segmentation on MRI scan slices. We trained (model info).

    It was trained on BraTS2020 dataset, which included MRI scan slices with segmentations performed manually by neuro-radiologists.

    User guide:
    * Root section redirects the user from http://127.0.0.1:8000 to http://127.0.0.1:8000/docs.

    * Section Health shows that the API is running. 

    * Steps to follow to get a prediction:
    1. Scroll to "Predict" section. 
    2. Click on the arrow pointing down.
    3. Click on "Try it out" button.
    4. Click on "Browse..." and select the scan file to be uploaded from your computer.
    5. Click on the blue button "Execute".
    6. Scroll to Response Body: you may visualize or save the '.png' file with the regions of the tumor from the uploaded scan. 
    
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
    version = "alpha",
)

@app.get("/", description = "Root endpoint")
async def root():
    return RedirectResponse(url='/docs')

@app.get("/health",
         response_model=HealthResponse,
         description="Check if the API is running.")
async def health():
    return {"message": "API is running"}

@app.post("/predict",
          description="Upload a '.h5' MRI scan slice. Returns a '.png' file with the segmentation mask with 3 regions of the tumor identified.")
async def predict(scan: UploadFile = File(...)):
    if not scan.filename.endswith(".h5"):
        raise HTTPException(
            status_code=400,
            detail="File must be '.h5'"
        )
    input_data = load_h5_input(scan)
    #preprocessed_input = preprocess(input_data)
    #prediction = model.predict(preprocessed_input)
    prediction = fake_prediction()
    user_image = segmentation_to_png(prediction)
    return StreamingResponse(
        user_image,
        media_type="image/png"
    )

def main():
    # model = ...
    create_random_file()

if __name__ == '__main__':
    main()
