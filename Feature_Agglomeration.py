import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from sklearn.decomposition import PCA
from define_params import DefineParams

class FeatureAgglomerationSegmenter(DefineParams):
    def __init__(self,file_name, image, n_clusters):
        self.image = image
        self.n_clusters = n_clusters
        self.resized_image=image
        self.original_shape = image.shape
        self.segmented_image=image
        self.file_name = file_name

    def resize_image(self, max_size=500):
        h, w, _ = self.image.shape
        if h > max_size and h <= max_size + 300:
            scale = max_size / max(h, w)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR_EXACT)
        elif h > max_size + 300:
            scale = (max_size+300) / (max(h, w)*5)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR_EXACT)
        return self.image

    def segment_algo(self):
        self.resized_image = self.resize_image()
        pixel_values = self.resized_image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        agglo = cluster.FeatureAgglomeration(n_clusters=self.n_clusters)
        agglo.fit(pixel_values.T)

        labels = agglo.labels_  
        return labels     

    def create_segmented_image(self,labels, color_palette):
        if labels is None:
            raise ValueError("L'algorithme Feature Agglomeration doit être appliqué avant de créer l'image segmentée.")

        segmented_image = np.zeros((self.resized_image.shape[0], self.resized_image.shape[1], 3), dtype=np.uint8)

        palette = plt.cm.get_cmap(color_palette, self.n_clusters)
        for i in range(self.n_clusters):
            color = np.array(palette(i)[:3])
            color = (color * 255).astype(np.uint8)
            segmented_image[labels.reshape(self.resized_image.shape[:2]) == i] = color

        segmented_image = cv2.resize(segmented_image, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_CUBIC)
        return segmented_image
    
    def color_image(self, image):
        img = self.file_name[0] + "_feature_agglo." + self.file_name[1]
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
