import numpy as np
from scipy import spatial

__all__ = [
    'KMeans'
]

class KMeans(object):

    def __init__(self, data, k=None, kmeans=None):
        self.x = np.atleast_2d(data)
        self.data_size = len(data)
        self.k = k
        self.m = kmeans
        self.clusters = np.empty(self.data_size)
        self.converged = False

    def init_means(self):
        self.m = self.x[
            np.random.choice(np.arange(self.data_size), self.k, False)]

    def distance(self, m):
        '''Calculate distance between each data point and m.

        Parameters
        ----------

        m : np.array
            vector, the data point to which the distance is measured.

        Returns
        -------
        np.ndarray
            vector of distance between each data point in x and m
        '''
        dist = spatial.distance.cdist(self.x, m)
        return dist

    def centroids(self, clusters):
        k = (np.max(clusters) + 1) if self.k is None else self.k
        shape = (k,) + self.x.shape[1:]
        results = np.empty(shape)
        for i in range(self.k):
            np.mean(self.x[clusters==i], axis=0, out=results[i])
        return results

    def fit(self, epoch):
        if self.m is None: self.init_means()
        iterations = 0
        while (iterations < epoch):
            # 1.
            # Calculate the distance between each point in self.x
            # and self.m
            distance = self.distance(self.m)
            # 2.
            # Assign self.x to according to the nearest distance
            new_clusters = np.argmin(distance, axis=1)
            new_means = self.centroids(new_clusters)
            # If the clusters stop changing, stop
            if np.array_equal(self.clusters, new_clusters):
                self.converged = True
                break
            self.clusters = new_clusters
            self.m = new_means
            iterations += 1
        return (self.converged, self.clusters)
