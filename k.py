import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import streamlit as st
from define_params import DefineParams
import plotly.graph_objects as go

class KMeansSegmenter(DefineParams):
    def __init__(self,file_name, image, clusters=5):
        self.image = image
        self.original_shape = image.shape
        self.clusters = clusters
        self.resized_image = image
        self.file_name = file_name
        #self.segmented_image=None

    def resize_image(self, max_size=2600):
        h, w, _ = self.image.shape
        if h > max_size and h <= max_size + 300:
            scale = max_size / (max(h, w)*2)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR_EXACT)
        elif h > max_size + 300:
            scale = (max_size+300) / (max(h, w)*5)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR_EXACT)
        return self.image
    
    def segment_algo(self):
        self.resized_image = self.resize_image()
        pixel_values = self.resized_image.reshape((-1, 3)).astype(np.float32)
        kmeans = KMeans(n_clusters=self.clusters)
        labels = kmeans.fit_predict(pixel_values)
        return labels

    def create_segmented_image(self, labels, color_palette):
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Ignorer le bruit
            n_noise = list(labels).count(-1)

            segmented_image = np.zeros_like(self.resized_image)

            normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())

            for i, label in enumerate(unique_labels):
                if label != -1:
                    color = plt.cm.get_cmap(color_palette)(normalized_labels[labels == label][0])[:3]
                    color = (np.array(color) * 255).astype(np.uint8)
                    segmented_image[labels.reshape(self.resized_image.shape[:2]) == label] = color
            segmented_image = cv2.resize(segmented_image, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_CUBIC)
            return segmented_image

    def color_image(self, image):
        img = self.file_name[0] + "_Kmeans." + self.file_name[1]
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
    
