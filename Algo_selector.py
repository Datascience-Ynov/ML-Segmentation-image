import streamlit as st

class AlgorithmSelector:
    def get_algorithm(self):
        return st.selectbox(
            "Choisissez l'algorithme de clustering:",
            options=[
                "KMeans", "k_means", "DBSCAN", "Agglomerative Clustering", "MiniBKMeans",
                "Fuzzy C-means", "SOM", "Birch", "Optics", "Bissecting_Kmeans",
                "Feature_Agglomeration", "HDBScan", "MeanShift", "SpectralBiClustering",
                "SpectralClustering", "Cluster_optics_dbscan", "EstimateBandwith",
                "KMeansPlusPlus", "Affinity_prop"
            ]
        )

    def get_n_clusters(self):
        return st.slider("Nombre de clusters", min_value=1, max_value=20, value=5)

    def get_dbscan_params(self):
        eps = st.slider("Valeur de eps", min_value=0.1, max_value=10.0, value=1.0)
        min_samples = st.slider("Nombre minimum d'échantillons", min_value=1, max_value=20, value=5)
        return eps, min_samples

    def get_color_palette(self):
        return st.selectbox(
            "Choisissez une palette de couleurs:",
            options=["viridis", "plasma", "inferno", "magma", "cividis"]
        )

    def fuzzy_params(self):
        cluster = st.slider("Nombre de clusters", min_value=1, max_value=20, value=5)
        scale = st.slider("Valeur de scale", min_value=1, max_value=100, value=50)
        m = st.slider("Valeur de m", min_value=1, max_value=100, value=2)
        error = st.slider("Valeur de error", min_value=0.001, max_value=1.0, value=0.05)
        max_iter = st.slider("Valeur de max_iter", min_value=1, max_value=150, value=100)
        return cluster, scale, m, error, max_iter

    def minibkmeans_params(self):
        n_clusters = st.slider("Nombre de clusters", min_value=1, max_value=20, value=5)
        batch_size = st.slider("Taille du batch", min_value=1, max_value=1000, value=100)
        return n_clusters, batch_size

    def affinity_propagation_params(self):
        damping = st.slider("Valeur de damping", min_value=0.5, max_value=1.0, value=0.7)
        preference = st.slider("Valeur de preference", min_value=-50, max_value=0, value=-30)
        max_iter = st.slider("Valeur de max_iter", min_value=200, max_value=1000, value=50)
        convergence_iter = st.slider("Valeur de convergence_iter", min_value=15, max_value=100, value=15)
        random_state = st.slider("Valeur de random_state", min_value=1, max_value=100, value=5)
        return preference, damping, max_iter, random_state, convergence_iter

    def som_params(self):
        cluster = st.slider("Nombre de clusters", min_value=1, max_value=20, value=5)
        grid_size = st.slider("Valeur de grid_size", min_value=1, max_value=100, value=10)
        sigma = st.slider("Valeur de sigma", min_value=0.1, max_value=10.0, value=1.0)
        learning_rate = st.slider("Valeur de learning_rate", min_value=0.1, max_value=10.0, value=2.60)
        max_iter = st.slider("Valeur de max_iter", min_value=1, max_value=100, value=100)
        return cluster, grid_size, sigma, learning_rate, max_iter
   
    def optics_params(self):
        min_samples = st.slider("Valeur de min_samples", min_value=1, max_value=100, value=5)
        xi = st.slider("Valeur de xi", min_value=0.05, max_value=1.0, value=0.05)
        min_cluster_size = st.slider("Valeur de min_cluster_size", min_value=0.05, max_value=1.0, value=0.05)
        return min_samples, xi, min_cluster_size
    
    def optics_dbscan_params(self):
        min_samples = st.slider("Valeur de min_samples", min_value=1, max_value=100, value=20)
        xi = st.slider("Valeur de xi", min_value=0.05, max_value=1.0, value=0.05)
        min_cluster_size = st.slider("Valeur de min_cluster_size", min_value=0.05, max_value=1.0, value=0.05)
        eps = st.slider("Valeur de eps", min_value=0.1, max_value=10.0, value=3.24)
        return min_samples, xi, min_cluster_size, eps

    def estimate_bandwith_params(self):
        sample_size = st.slider("Valeur de sample_size", min_value=1, max_value=10000, value=500)
        quantile = st.slider("Valeur de quantile", min_value=0.1, max_value=1.0, value=0.3)
        return sample_size, quantile

    def agglo_params(self):
        n_clusters = st.slider("Nombre de clusters", min_value=1, max_value=20, value=5)
        linkage = st.selectbox("Type de linkage", options=["ward", "complete", "average", "single"])
        metric = st.selectbox("Métrique de distance", options=["euclidean", "l1", "l2", "manhattan", "cosine", "cityblock"])
        return n_clusters, linkage, metric

    def spectral_params(self):
        n_clusters = st.slider("Nombre de clusters", min_value=1, max_value=20, value=5)
        neighbors = st.slider("Nombre de voisins", min_value=1, max_value=100, value=10)
        affinity = st.selectbox("Choisissez une affinity:", options=["nearest_neighbors", "rbf"])
        assign_labels = st.selectbox("Choisissez une méthode d'assignation des labels:", options=["kmeans", "discretize"])
        return n_clusters, neighbors, affinity, assign_labels

    def min_cluster_size(self):
        min_cluster = st.slider("Valeur de min_cluster_size", min_value=1, max_value=100, value=5)
        min_samples = st.slider("Valeur de min_samples", min_value=1, max_value=100, value=5)
        return min_cluster, min_samples
    
    def get_bandwidth(self):
        bandwidth = st.slider("Valeur de bandwidth", min_value=0.1, max_value=10.0, value=1.0)
        return bandwidth