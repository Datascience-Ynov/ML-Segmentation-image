import numpy as np
import cv2
import matplotlib.pyplot as plt
from minisom import MiniSom
from skimage import color
import streamlit as st
from define_params import DefineParams

class SOMSegmenter(DefineParams):
    def __init__(self, file_name, image, grid_size=(10, 10), sigma=1.0, learning_rate=2.60, max_iter=100):
        if not isinstance(grid_size, tuple):
            raise ValueError("grid_size must be a tuple (rows, cols)")
        self.image = image
        self.grid_size = grid_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.resized_image = None
        self.som = None
        self.labels = None
        self.original_shape = image.shape
        self.colored_image = image
        self.file_name = file_name

    def resize_image(self, max_size=200):
        h, w, _ = self.image.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)))
        return self.image

    def segment_algo(self):
        lab_image = color.rgb2lab(self.image)

        data = lab_image.reshape(-1, 3)
        data = data / np.max(data)

        self.som = MiniSom(
            x=self.grid_size[0],
            y=self.grid_size[1],
            input_len=3,
            sigma=self.sigma,
            learning_rate=self.learning_rate
        )

        self.som.random_weights_init(data)

        self.som.train_random(data, self.max_iter)

        bmu_indices = np.array([self.som.winner(d) for d in data])
        self.labels = bmu_indices[:, 0] * self.grid_size[1] + bmu_indices[:, 1]
        return self.labels

    def create_segmented_image(self,labels, color_palette="viridis"):
        if self.labels is None or self.som is None:
            raise ValueError("L'algorithme SOM doit être appliqué avant de créer l'image segmentée.")

        segmented_image = self.labels.reshape(self.image.shape[:2])

        self.colored_image = np.zeros((self.image.shape[0], self.image.shape[1], 3))

        palette = plt.cm.get_cmap(color_palette, self.grid_size[0] * self.grid_size[1])
        for i in range(self.grid_size[0] * self.grid_size[1]):
            color = np.array(palette(i)[:3])
            self.colored_image[segmented_image == i] = color

        self.colored_image = cv2.resize(self.colored_image, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_AREA)        
        return self.colored_image

    def color_image(self, image):
        img = self.file_name[0] + "_som." + self.file_name[1]
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
