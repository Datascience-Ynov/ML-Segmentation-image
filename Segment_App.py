import streamlit as st
from Image_upload import ImageUploader
from Algo_selector import AlgorithmSelector
from Segmenter_factory import SegmenterFactory

class ImageSegmenterApp:
    def __init__(self):
        self.image_uploader = ImageUploader()
        self.algorithm_selector = AlgorithmSelector()
        self.segmenter_factory = SegmenterFactory()
        self.labels = None
        self.seg_image = None

    def run(self):
        """Exécute l'application."""
        uploaded_file = self.image_uploader.upload_image()
        if uploaded_file is not None:
            image = self.image_uploader.load_image(uploaded_file)
            self.image_uploader.show_original_image(image)

            algorithm = self.algorithm_selector.get_algorithm()
            color_palette = self.algorithm_selector.get_color_palette()

            # Récupérer les paramètres spécifiques à l'algorithme
            params = {}
            if algorithm == "KMeans":
                params["n_clusters"] = self.algorithm_selector.get_n_clusters()

            elif algorithm == "DBSCAN":
                params["eps"], params["min_samples"] = self.algorithm_selector.get_dbscan_params()
            
            elif algorithm == "Fuzzy C-means":
                params["cluster"], params["scale"], params["m"], params["error"], params["max_iter"] = (
                    self.algorithm_selector.fuzzy_params()
                )
            
            elif algorithm == "MiniBKMeans":
                params["n_clusters"], params["batch_size"] = self.algorithm_selector.minibkmeans_params()
            
            elif algorithm == "Affinity_prop":
                params["preference"], params["damping"], params["max_iter"], params["random_state"], params["convergence_iter"] = (
                    self.algorithm_selector.affinity_propagation_params()
                )
            
            elif algorithm == "SOM":
                params["cluster"], params["grid_size"], params["sigma"], params["learning_rate"], params["max_iter"] = (
                    self.algorithm_selector.som_params()
                )
            
            elif algorithm == "MeanShift":
                params["bandwidth"] = self.algorithm_selector.get_bandwidth()
            
            elif algorithm == "SpectralBiClustering":
                params["n_clusters"] = self.algorithm_selector.get_n_clusters()
            
            elif algorithm == "SpectralClustering":
                params["n_clusters"], params["neighbors"], params["affinity"], params["assign_labels"] = (
                    self.algorithm_selector.spectral_params()
                )
            
            elif algorithm == "HDBScan":
                params["min_cluster_size"], params["min_samples"] = self.algorithm_selector.min_cluster_size()
            
            elif algorithm == "Bissecting_Kmeans":
                params["n_clusters"] = self.algorithm_selector.get_n_clusters()
            
            elif algorithm == "Feature_Agglomeration":
                params["n_clusters"] = self.algorithm_selector.get_n_clusters()
            
            elif algorithm == "Optics":
                params["min_samples"], params["xi"], params["min_cluster_size"] = self.algorithm_selector.optics_params()
            
            elif algorithm == "Cluster_optics_dbscan":
                params["min_samples"], params["xi"], params["min_cluster_size"], params["eps"] = (
                    self.algorithm_selector.optics_dbscan_params()
                )
            
            elif algorithm == "EstimateBandwith":
                params["sample_size"], params["quantile"] = self.algorithm_selector.estimate_bandwith_params()
            
            elif algorithm == "KMeansPlusPlus":
                params["n_clusters"] = self.algorithm_selector.get_n_clusters()
            
            elif algorithm == "Agglomerative Clustering":
                params["n_clusters"], params["linkage"], params["metric"] = self.algorithm_selector.agglo_params()

            elif algorithm == "Birch":
                params["n_clusters"] = self.algorithm_selector.get_n_clusters()
            
            elif algorithm == "k_means":
                params["n_clusters"] = self.algorithm_selector.get_n_clusters()

            file_name = uploaded_file.name.split(".")
            segmenter = self.segmenter_factory.create_segmenter(algorithm, file_name, image, **params)

            if st.button("Segmenter l'image"):
                self.labels = segmenter.segment_algo()
                self.seg_image = segmenter.create_segmented_image(self.labels, color_palette)
                st.image(self.seg_image, caption=f'Image Segmentée avec {algorithm}', use_container_width=True)
        
            if st.button("Save Image"):
                self.labels = segmenter.segment_algo()
                segmented_image = segmenter.create_segmented_image(self.labels, color_palette)
                segmenter.color_image(segmented_image)

            if st.button("Plot 3D clusters"):
                segmenter.plot_3d(self.labels, color_palette)

