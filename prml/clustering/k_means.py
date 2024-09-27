import numpy as np
from scipy.spatial.distance import cdist


class KMeans(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, x, iter_max=100):
        """
        perform k-means algorithm

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input data
        iter_max : int
            maximum number of iterations

        Returns
        -------
        centers : (n_clusters, n_features) ndarray
            center of each cluster
        """
        eye = np.eye(self.n_clusters)
        centers = x[np.random.choice(len(x), self.n_clusters, replace=False)]
        for _ in range(iter_max):
            prev_centers = np.copy(centers)
            D = cdist(x, centers)
            cluster_index = np.argmin(D, axis=1)
            cluster_index = eye[cluster_index]
            centers = np.sum(x[:, None, :] * cluster_index[:, :, None], axis=0) / np.sum(cluster_index, axis=0)[:, None]
            if np.allclose(prev_centers, centers):
                break
        self.centers = centers

    def predict(self, x):
        """
        calculate closest cluster center index

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        index : (sample_size,) ndarray
            indicates which cluster they belong
        """
        D = cdist(x, self.centers)
        return np.argmin(D, axis=1)
