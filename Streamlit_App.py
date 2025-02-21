import streamlit as st
from Segment_App import ImageSegmenterApp

class StreamlitApp:
    def __init__(self):
        st.title("Segmentation d'Image")
        self.image_segmenter_app = ImageSegmenterApp()

    def run(self):
        """Exécute l'application."""
        self.image_segmenter_app.run()

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()

# streamlit run Streamlit_App.py