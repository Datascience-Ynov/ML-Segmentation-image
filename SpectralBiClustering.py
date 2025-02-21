from sklearn.cluster import SpectralBiclustering
import numpy as np
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from define_params import DefineParams

class SpectralBiClusteringSegmenter(DefineParams):
    def __init__(self, file_name, image, n_clusters):
        self.image = image
        self.original_shape = image.shape
        self.n_clusters = n_clusters
        self.resized_image = image
        self.segmented_image = image
        self.file_name = file_name

    def resize_image(self, max_size=200):
        h, w, _ = self.image.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)))
        return self.image

    def segment_algo(self):
        pixel_values = self.image.reshape((-1, 3)).astype(np.float32)
        
        if self.n_clusters > pixel_values.shape[0]:
            raise ValueError("n_clusters should be less than or equal to the number of samples")
        
        bicluster = SpectralBiclustering(n_clusters=self.n_clusters)
        bicluster.fit(pixel_values)
        labels = bicluster.row_labels_
        return labels.reshape(self.image.shape[:2])


    def create_segmented_image(self, labels, color_palette):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Ignorer le bruit

        self.segmented_image = np.zeros_like(self.image)

        if labels.max() == labels.min():
            normalized_labels = labels
        else:
            normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())

        labels_reshaped = labels.reshape(self.image.shape[:2])
        for i, label in enumerate(unique_labels):
            if label != -1:
                color = plt.cm.get_cmap(color_palette)(normalized_labels[labels == label][0])[:3]
                color = (np.array(color) * 255).astype(np.uint8)
                self.segmented_image[labels_reshaped == label] = color

        self.segmented_image = cv2.resize(self.segmented_image, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_CUBIC)
        return self.segmented_image

    def color_image(self, image):
        img = self.file_name[0] + "_spec_Bi_clus." + self.file_name[1]
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
