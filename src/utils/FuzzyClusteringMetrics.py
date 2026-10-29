import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist

class FuzzyClusteringMetrics:
    def __init__(self, X, u, centers, m=2):
        """
        X: (n_samples, n_features) - Data points
        u: (n_clusters, n_samples) - Fuzzy membership matrix from cmeans
        centers: (n_clusters, n_features) - Cluster centroids
        m: float - Fuzziness coefficient (usually 2)
        """
        self.X = X
        self.u = u
        self.centers = centers
        self.m = m
        self.n_samples = X.shape[0]
        self.n_clusters = centers.shape[0]
        
        # Derived Hard Labels for sklearn metrics
        self.hard_labels = np.argmax(u, axis=0)
        # Store unique labels count to prevent sklearn ValueErrors
        self.n_unique_labels = len(np.unique(self.hard_labels))
        
        # Performance Optimization: Cache pairwise distances
        # This prevents redundant O(N*C) calculations in multiple metrics
        self.dists_sq = cdist(self.X, self.centers, metric='sqeuclidean')
        self.dists = np.sqrt(self.dists_sq)

    # --- HARD METRICS (Scikit-Learn Wrappers & Custom) ---

    def get_silhouette(self):
        if self.n_unique_labels < 2: return None
        return silhouette_score(self.X, self.hard_labels)

    def get_calinski_harabasz(self):
        if self.n_unique_labels < 2: return None
        return calinski_harabasz_score(self.X, self.hard_labels)

    def get_davies_bouldin(self):
        if self.n_unique_labels < 2: return None
        return davies_bouldin_score(self.X, self.hard_labels)

    def get_distortion(self):
        # Weighted Sum of Squared Errors
        # Apply fuzzy weights (u^m) to pre-calculated squared distances
        return np.sum((self.u ** self.m).T * self.dists_sq)

    def get_dunn_index(self):
        # Simplified Dunn: Min inter-cluster dist / Max intra-cluster dist
        if self.n_clusters < 2: return None
        
        # Distances between centroids
        center_dists = cdist(self.centers, self.centers)
        np.fill_diagonal(center_dists, np.inf)
        min_separation = np.min(center_dists)
        
        # Max diameter of clusters (approximation using distance to center)
        # Optimized to use the cached distance matrix
        max_diameter = 0
        for c in range(self.n_clusters):
            # Fetch pre-calculated distances from points in cluster 'c' to center 'c'
            cluster_dists = self.dists[self.hard_labels == c, c]
            if len(cluster_dists) > 0:
                # Max distance from center * 2 is a rough diameter proxy
                d = np.max(cluster_dists)
                max_diameter = max(max_diameter, d * 2)
        
        if max_diameter == 0: return 0
        return min_separation / max_diameter

    # --- FUZZY METRICS (Manual Implementation) ---

    def get_fpc(self):
        # Fuzzy Partition Coefficient
        return np.sum(self.u ** 2) / self.n_samples

    def get_mpc(self):
        # Modified Partition Coefficient
        if self.n_clusters < 2: return None  # Prevent ZeroDivisionError
        fpc = self.get_fpc()
        c = self.n_clusters
        return 1 - (c / (c - 1)) * (1 - fpc)

    def get_fpe(self):
        # Fuzzy Partition Entropy
        # Avoid log(0)
        u_safe = np.clip(self.u, 1e-10, 1.0) 
        return -np.sum(self.u * np.log(u_safe)) / self.n_samples

    def get_xie_beni(self):
        # Compactness / Separation
        # Numerator: Compactness uses fuzzifier m and cached distances
        compactness = np.sum((self.u ** self.m).T * self.dists_sq)
        
        # Denominator: Separation (Min squared distance between centers)
        center_dists_sq = cdist(self.centers, self.centers, metric='sqeuclidean')
        np.fill_diagonal(center_dists_sq, np.inf)
        min_sep = np.min(center_dists_sq)
        
        if min_sep == 0: return np.inf
        return compactness / (self.n_samples * min_sep)

    def get_fukuyama_sugeno(self):
        # J_m - K_m (Difference between local and global compactness)
        mean_X = np.mean(self.X, axis=0).reshape(1, -1)
        
        # Use cached squared distances
        term1 = np.sum((self.u ** self.m).T * self.dists_sq)
        
        center_dists_global = cdist(self.centers, mean_X, metric='sqeuclidean')
        # Sum of u_ik^m * ||v_i - v_bar||^2
        # Note: Sum of u_ik over samples is the 'cardinality' of the cluster
        cardinality = np.sum(self.u ** self.m, axis=1)
        term2 = np.sum(cardinality.reshape(-1, 1) * center_dists_global)
        
        return term1 - term2

    def calculate_all(self):
        return {
            'c': self.n_clusters,
            'Distortion': self.get_distortion(),
            'Silhouette': self.get_silhouette(),
            'Calinski_Harabasz': self.get_calinski_harabasz(),
            'Davies_Bouldin': self.get_davies_bouldin(),
            'Dunn': self.get_dunn_index(),
            'FPC': self.get_fpc(),
            'MPC': self.get_mpc(),
            'FPE': self.get_fpe(),
            'Xie_Beni': self.get_xie_beni(),
            'Fukuyama_Sugeno': self.get_fukuyama_sugeno()
        }