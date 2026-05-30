import requests

import streamlit as st


def main():
    """
    Demo for the users.
    """
    API_URL = "http://127.0.0.1:8000/predict"

    st.set_page_config(
        page_title="Brain Tumor Segmentation Demo", layout="centered"
    )
    st.markdown(
        "<h1 style='color: pink'>Brain Tumor Segmentation Demo<h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='color:pink'>Upload a '.h5' MRI scan slice. Returns a '.png' file with the segmentation mask with 3 regions of the tumor identified. Our model performs segmentation on MRI scan slices. We trained (model info). It was trained on BraTS2020 dataset, which included MRI scan slices with segmentations performed manually by neuro-radiologists. </h3>",
        unsafe_allow_html=True,
    )
    uploading = st.file_uploader("Upload MRI scan '.h5'...", type=["h5"])
    if uploading != None:
        st.success("File uploaded! :D")
        if st.button("Run Segmentation"):
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
                st.error(
                    f"Sorry, prediction failed: {response.status_code}. :("
                )
