import uvicorn

import numpy as np
import h5py
import io
from PIL import Image

from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

# import model

class HealthResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    detail: str

#class Prediction(BaseModel):

def create_random_file():
    data = np.random.rand(256, 256)
    with h5py.File("test_api.h5", "w") as f:
        f.create_dataset("scan", data=data)

def fake_prediction():
    mask = np.zeros((256, 256))
    mask[50:100, 50:100] = 1
    return mask

def load_h5_input(file: UploadFile = File(...)) -> np.ndarray:
    """
    Extracts data from a h5 file.

    If the file is not h5, raise error.
    """
    try:
        with h5py.File(file.file, "r") as f:
            key = list(f.keys())[0]
            if len(key) == 0:
                raise HTTPException(status_code=400, detail="No scan found in this file")
            input_data = np.array(f[key])
            return input_data 
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid '.h5' file.")
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Invalid '.h5' file: {str(e)}")

def preprocess(data: np.ndarray) -> np.ndarray:
    return data

def segmentation_to_png(prediction: np.ndarray):
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
    
    Input 
    Upload a `.h5' file with a slice from a MRI scan.

    Output
    Segmentation mask as a `.png` file:
    - red, green, blue = regions of the tumor
    - black = background

    Important!
    This API is for educational and testing purposes only.
    This is not a diagnostic tool!
    
    """,
    version = "alpha",
)

@app.get("/",
        description="Check if the API is running.")
async def root():
    return {"message": "Brain Tumor Segmentation Classifier is running"}

# predictions: work in progress
@app.post("/predict")

async def predict(scan: UploadFile):
    if not scan.filename.endswith(".h5"):
        raise HTTPException(
            status_code=400,
            detail="File must be '.h5'"
        )
    input_data = load_h5_input(scan)
    preprocessed_input = preprocess(input_data)
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


    
    uvicorn.run(app,
                host = "127.0.0.1",
                port = 8000,
                reload = True)

if __name__ == '__main__':
    main()