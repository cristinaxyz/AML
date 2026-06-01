import io

import requests
from PIL import Image

import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Brain Tumor Segmentation Demo", layout="centered"
)
st.markdown(
    "<h1 style='color: pink'>Brain Tumor Segmentation Demo<h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """<h3 style='color: red; margin-bottom : 0';>Important<h3>
    <h4 style='color: red; margin-bottom : 0;'>This API is for educational and testing purposes only.<h4>
    <h4 style='color: red; margin-bottom : 0;'>This is not a diagnostic tool!<h4>
    <h4 style='color:pink; margin-top : 5;'>Our model performs segmentation on MRI scan slices. We trained a UNet model to perform segmentation. The model was trained on BraTS2020 dataset, which included MRI scan slices with segmentations performed manually by neuro-radiologists. </h4>""",
    unsafe_allow_html=True,
)

st.write(
    "User's guide\n"
    "1. Click on the button <Upload> below.\n"
    "2. Select a '.h5' file from your computer.\n"
    "3. Click on the button <Run Segmentation>.\n"
    "4. Click on the button <Download '.png!'> if you want to save the prediction in the '.png' format.\n"
    "5. Once you visualized/saved the prediction, you may click on <X> in the right corner of the file you uploaded and start again from step 1 for a new prediction!"
)

uploading = st.file_uploader("Upload MRI scan '.h5'...", type=["h5"])
if uploading is not None:
    st.success("File uploaded! :D")
    if st.button("Run Segmentation"):
        with st.spinner("Running the model..."):
            files = {"scan": (uploading.name, uploading.getvalue())}
            response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            st.success("Prediction is done!")
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption="Segmentation Mask", width="stretch")
            st.download_button(
                label="Download '.png'!",
                data=response.content,
                file_name="segmentation.png",
            )
        else:
            st.error(f"Sorry, prediction failed: {response.status_code}. :(")
