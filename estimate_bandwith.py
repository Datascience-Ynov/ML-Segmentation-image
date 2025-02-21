import numpy as np
from sklearn.cluster import estimate_bandwidth, MeanShift
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from sklearn.cluster import OPTICS
from define_params import DefineParams

class EstimateBandwithSegmenter(DefineParams):
    def __init__(self, file_name,image, sample_size, quantile):
        self.image = image
        self.original_shape = image.shape
        self.resized_image = image
        self.sample_size = sample_size
        self.quantile = quantile
        self.segmented_image = image
        self.file_name = file_name

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
        pixel_values = self.resized_image.reshape((-1, 3))

        bandwidth = estimate_bandwidth(pixel_values, quantile=self.quantile, n_samples=self.sample_size)
        
        meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift_model.fit(pixel_values)
        labels = meanshift_model.labels_
        return labels

    def create_segmented_image(self, labels, color_palette):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        segmented_image = np.zeros_like(self.resized_image)

        if labels.max() == labels.min():
            normalized_labels = labels
        else:
            normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())

        labels_reshaped = labels.reshape(self.resized_image.shape[:2])
        for i, label in enumerate(unique_labels):
            if label != -1:
                color = plt.cm.get_cmap(color_palette)(normalized_labels[labels == label][0])[:3]
                color = (np.array(color) * 255).astype(np.uint8)
                segmented_image[labels_reshaped == label] = color
        segmented_image = cv2.resize(segmented_image, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_CUBIC)
        return segmented_image
    
    def color_image(self, image):
        img = self.file_name[0] + "_estim_band." + self.file_name[1]
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
