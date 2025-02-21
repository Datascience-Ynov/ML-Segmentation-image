import streamlit as st
import cv2
import numpy as np

class ImageUploader:
    def upload_image(self):
        """Permet à l'utilisateur de télécharger une image."""
        return st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    def load_image(self, uploaded_file):
        """Charge et convertit l'image en RGB."""
        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.write("Image chargée et convertie en RGB.")
            return image
        return None

    def show_original_image(self, image):
        """Affiche l'image originale."""
        st.image(image, caption='Image Originale', use_container_width=True)
        