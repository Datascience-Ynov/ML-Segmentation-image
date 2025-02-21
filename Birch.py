import cv2
import numpy as np
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import streamlit as st
from define_params import DefineParams

class BirchSegmenter(DefineParams):
    def __init__(self, file_name, image, n_clusters, threshold=0.5):
        self.image = image
        self.original_shape = image.shape
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.resized_image = image
        self.segmented_image = image
        self.file_name = file_name

    def resize_image(self, max_size=500):
        h, w, _ = self.image.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)))
        return self.image

    def segment_algo(self):
        scale_percent = 50
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        self.resized_image = cv2.resize(self.image, (width, height))
        
        pixel_values = self.resized_image.reshape((-1, 3)).astype(np.float32) / 255.0
        
        birch_model = Birch(n_clusters=self.n_clusters, threshold=min(self.threshold, 0.1))
        labels = birch_model.fit_predict(pixel_values)
        
        return labels
    
    def create_segmented_image(self, labels, color_palette):
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Ignorer le bruit
            n_noise = list(labels).count(-1)

            segmented_image = np.zeros_like(self.resized_image)

            if labels.max() == labels.min():
                normalized_labels = labels
            else:
                normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())

            for i, label in enumerate(unique_labels):
                if label != -1:
                    color = plt.cm.get_cmap(color_palette)(normalized_labels[labels == label][0])[:3]  # Ignorer l'alpha
                    color = (np.array(color) * 255).astype(np.uint8)  # Convertir en valeurs RGB (0-255)
                    segmented_image[labels.reshape(self.resized_image.shape[:2]) == label] = color
            
            return segmented_image
    
    def color_image(self, image):
        img = self.file_name[0] + "_Birch." + self.file_name[1]
        image_final = "results/" + img
        cv2.imwrite(f"{image_final}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        st.write(f"{image_final} saved successfully.")

    def plot_3d(self, labels, color_palette):        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        pixel_values = self.resized_image.reshape((-1, 3)).astype(np.float32)
        ax.scatter(pixel_values[:, 0], pixel_values[:, 1], pixel_values[:, 2], c=labels, cmap=color_palette, marker='o', edgecolor='k', s=50)
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        st.pyplot(fig)
