
class SegmenterFactory:
    def create_segmenter(self, algorithm, file_name, image, **kwargs):
        """Crée une instance de segmenter en fonction de l'algorithme choisi."""
        if algorithm == "KMeans":
            from k import KMeansSegmenter
            return KMeansSegmenter(file_name, image, kwargs.get("n_clusters"))
        
        elif algorithm == "k_means":
            from k_means import K_meansSegmenter
            return K_meansSegmenter(file_name, image, kwargs.get("n_clusters"))
        
        elif algorithm == "DBSCAN":
            from db import DBSCANSegmenter
            return DBSCANSegmenter(file_name, image, kwargs.get("eps"), kwargs.get("min_samples"))
        
        elif algorithm == "Fuzzy C-means":
            from Fuzzy_C_means import FuzzyCMeansSegmenter
            return FuzzyCMeansSegmenter(
                file_name, image, kwargs.get("cluster"), kwargs.get("scale"),
                kwargs.get("m"), kwargs.get("error"), kwargs.get("max_iter")
            )
        
        elif algorithm == "MiniBKMeans":
            from MiniBKmeans import MiniBatchKMeansSegmenter
            return MiniBatchKMeansSegmenter(file_name, image, kwargs.get("n_clusters"), kwargs.get("batch_size"))
        
        elif algorithm == "Affinity_prop":
            from Affinity_propagation import AffinityPropagationSegmenter
            return AffinityPropagationSegmenter(
                file_name, image, kwargs.get("preference"), kwargs.get("damping"),
                kwargs.get("max_iter"), kwargs.get("random_state"), kwargs.get("convergence_iter")
            )
        
        elif algorithm == "SOM":
            from SOM import SOMSegmenter
            return SOMSegmenter(
                file_name, image, grid_size=(kwargs.get("grid_size"), kwargs.get("grid_size")),
                sigma=kwargs.get("sigma"), learning_rate=kwargs.get("learning_rate"),
                max_iter=kwargs.get("max_iter")
            )
        
        elif algorithm == "Birch":
            from Birch import BirchSegmenter
            return BirchSegmenter(file_name, image, kwargs.get("n_clusters"))
        
        
        elif algorithm == "MeanShift":
            from MeanShift import MeanShiftSegmenter
            return MeanShiftSegmenter(file_name, image, kwargs.get("bandwidth"))
        
        elif algorithm == "SpectralBiClustering":
            from SpectralBiClustering import SpectralBiClusteringSegmenter
            return SpectralBiClusteringSegmenter(file_name, image, kwargs.get("n_clusters"))
        
        elif algorithm == "SpectralClustering":
            from SpectralClustering import SpectralClusteringSegmenter
            return SpectralClusteringSegmenter(
                file_name, image, kwargs.get("n_clusters"), kwargs.get("neighbors"),
                kwargs.get("affinity"), kwargs.get("assign_labels")
            )
        
        elif algorithm == "HDBScan":
            from HDBScan import HDBScanSegmenter
            return HDBScanSegmenter(file_name, image, kwargs.get("min_cluster_size"), kwargs.get("min_samples"))
        
        elif algorithm == "Bissecting_Kmeans":
            from Bissecting_Kmeans import BissectingKMeansSegmenter
            return BissectingKMeansSegmenter(file_name, image, kwargs.get("n_clusters"))
        
        elif algorithm == "Feature_Agglomeration":
            from Feature_Agglomeration import FeatureAgglomerationSegmenter
            return FeatureAgglomerationSegmenter(file_name, image, kwargs.get("n_clusters"))
        
        elif algorithm == "Optics":
            from Optics import OpticsSegmenter
            return OpticsSegmenter(file_name, image, kwargs.get("min_samples"), kwargs.get("xi"), kwargs.get("min_cluster_size"))
        
        elif algorithm == "Cluster_optics_dbscan":
            from Cluster_optics_dbscan import ClusterOpticsDbscanSegmenter
            return ClusterOpticsDbscanSegmenter(
                file_name, image, kwargs.get("min_samples"), kwargs.get("xi"),
                kwargs.get("min_cluster_size"), kwargs.get("eps")
            )
        
        elif algorithm == "EstimateBandwith":
            from estimate_bandwith import EstimateBandwithSegmenter
            return EstimateBandwithSegmenter(file_name, image, kwargs.get("sample_size"), kwargs.get("quantile"))
        
        elif algorithm == "KMeansPlusPlus":
            from k_means_plus_plus import KMeansPlusPlusSegmenter
            return KMeansPlusPlusSegmenter(file_name, image, kwargs.get("n_clusters"))
        
        elif algorithm == "Agglomerative Clustering":
            from agglo import AgglomerativeSegmenter
            return AgglomerativeSegmenter(
                file_name, image, kwargs.get("n_clusters"), kwargs.get("linkage"),
                kwargs.get("metric")
            )

        else:
            raise ValueError(f"Algorithme non supporté : {algorithm}")