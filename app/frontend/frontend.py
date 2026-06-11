import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/post_images"

st.title("Chest X-Ray Disease Detection")

image = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if image is not None:

    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):

        files = {
            "upload": (
                image.name,
                image.getvalue(),
                image.type
            )
        }

        response = requests.post(
            API_URL,
            files=files
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                f"Prediction: {result['Predicted_class']}"
            )

            st.write(
                f"Confidence: {result['Confidence']:.2%}"
            )

        else:
            st.error(
                f"API Error: {response.text}"
            )