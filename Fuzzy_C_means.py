import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import streamlit as st
from skimage.color import rgb2gray
import skimage.color as color
from define_params import DefineParams


class FuzzyCMeansSegmenter(DefineParams):
    def __init__(self, file_name, image, n_clusters, scale_percent, m=2, error=0.05, max_iter=100):
        self.image = image
        self.original_shape = image.shape
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.max_iter = max_iter
        self.resized_image = None
        self.scale_percent = scale_percent
        self.segmented_image = image
        self.file_name = file_name

    def resize_image(self):
        gray_image = rgb2gray(self.image)
        width = int(gray_image.shape[1] * self.scale_percent / 100)
        height = int(gray_image.shape[0] * self.scale_percent / 100)
        resized_image = cv2.resize(gray_image, (width, height), interpolation=cv2.INTER_LINEAR_EXACT)

        return resized_image

    def segment_algo(self):
        
        self.resized_image = self.resize_image()
        pixel_values = self.resized_image.reshape(-1, 1)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            pixel_values.T, self.n_clusters, self.m, error=self.error, maxiter=self.max_iter, init=None
        )
        labels = np.argmax(u, axis=0)
        return labels

    def create_segmented_image(self, labels, color_palette):

        if self.resized_image is None or len(self.resized_image.shape) != 2:
            raise ValueError("self.resized_image doit être une image 2D (niveaux de gris).")

        segmented_image = labels.reshape(self.resized_image.shape)

        colored_image = np.zeros((self.resized_image.shape[0], self.resized_image.shape[1], 3))

        palette = plt.cm.get_cmap(color_palette, self.n_clusters)
        for i in range(self.n_clusters):
            color = np.array(palette(i)[:3])
            colored_image[segmented_image == i] = color

        colored_image = cv2.resize(colored_image, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return colored_image

    def color_image(self, image):
        img = self.file_name[0] + "_fuzzy." + self.file_name[1]
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
