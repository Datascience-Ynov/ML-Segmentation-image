from sklearn.cluster import SpectralClustering, SpectralCoclustering
import numpy as np
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import spectral_clustering
from define_params import DefineParams

class SpectralClusteringSegmenter(DefineParams):
    def __init__(self,file_name, image, n_clusters ,neighbors, affinity, assign_labels):
        self.image = image
        self.original_shape = image.shape
        self.n_clusters = n_clusters
        self.neighbors = neighbors
        self.affinity = affinity
        self.assign_labels = assign_labels
        self.segmented_image = image
        self.file_name = file_name
        self.resized_image = image

    def resize_image(self, max_size=200):
        h, w, _ = self.image.shape
        if h > max_size and h <= max_size + 300:
            scale = max_size / max(h, w)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR_EXACT)
        elif h > max_size + 300:
            scale = (max_size+300) / (max(h, w)*7)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR_EXACT)
        return self.image
    
    def segment_algo(self):
        self.resized_image = self.resize_image()
        pixel_values = self.resized_image.reshape((-1, 3)).astype(np.float32) / 255.0
        
        spectral = SpectralClustering(n_clusters=self.n_clusters, affinity=self.affinity,
                n_neighbors=self.neighbors, assign_labels=self.assign_labels, 
                random_state=42)
        labels = spectral.fit_predict(pixel_values)

        return labels.reshape(self.resized_image.shape[:2])

    def create_segmented_image(self, labels, color_palette):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        self.segmented_image = np.zeros_like(self.resized_image)

        if labels.max() == labels.min():
            normalized_labels = labels
        else:
            normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())

        labels_reshaped = labels.reshape(self.resized_image.shape[:2])
        for i, label in enumerate(unique_labels):
            if label != -1:
                color = plt.cm.get_cmap(color_palette)(normalized_labels[labels == label][0])[:3]
                color = (np.array(color) * 255).astype(np.uint8)
                self.segmented_image[labels_reshaped == label] = color

        self.segmented_image = cv2.resize(self.segmented_image, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_NEAREST)

        return self.segmented_image
    
    def color_image(self, image):
        img = self.file_name[0] + "_spec_clus." + self.file_name[1]
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
