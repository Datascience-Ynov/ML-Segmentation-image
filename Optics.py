import cv2
import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import streamlit as st
from define_params import DefineParams

class OpticsSegmenter(DefineParams):
    def __init__(self,file_name, image,min_samples=5, xi=0.05, min_cluster_size=0.05):
        self.image = image
        self.original_shape = image.shape
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.resized_image = image
        self.segmented_image = image
        self.file_name = file_name

    def resize_image(self):
        scale_percent = 10 if self.image.shape[0] > 500 else 50
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        self.resized_image = cv2.resize(self.image, (width, height), interpolation=cv2.INTER_LINEAR_EXACT)
        pixel_values = self.resized_image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        return pixel_values

    def segment_algo(self):
        
        pixel_values = self.resize_image()
        optics_model=OPTICS(min_samples=int(self.min_samples), xi=self.xi, min_cluster_size=self.min_cluster_size)
        optics_model.fit(pixel_values)
        labels = optics_model.labels_
        return labels

    def create_segmented_image(self, labels, color_palette):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Ignorer le bruit

        self.segmented_image = np.zeros_like(self.resized_image)

        if labels.max() == labels.min():
            normalized_labels = labels
        else:
            normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())

        labels_reshaped = labels.reshape(self.resized_image.shape[:2])
        for i, label in enumerate(unique_labels):
            if label != -1:
                color = plt.cm.get_cmap(color_palette)(normalized_labels[labels == label][0])[:3]  # Ignorer l'alpha
                color = (np.array(color) * 255).astype(np.uint8)
                self.segmented_image[labels_reshaped == label] = color
        
        self.segmented_image = cv2.resize(self.segmented_image, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_CUBIC)
        return self.segmented_image
    
    def color_image(self, image):
        img = self.file_name[0] + "_optics." + self.file_name[1]
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
