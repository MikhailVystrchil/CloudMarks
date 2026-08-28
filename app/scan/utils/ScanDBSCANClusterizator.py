import numpy as np
from sklearn.cluster import DBSCAN


class ScanDBSCANClusterizator:

    def __init__(self, scan):
        self.scan = scan

    def __scan_to_numpy(self):
        return np.array([[p.x, p.y, p.z] for p in self.scan])

    def compute_clusters(self, *args, eps=0.01, min_samples=10, **kwargs):
        labels = self.__dbscan_cluster_points(eps=eps, min_samples=min_samples)
        for p, n in zip(self.scan, labels):
            setattr(p, "labels", n)
        return labels

    def __dbscan_cluster_points(self, eps=0.01, min_samples=10):
        X = self.__scan_to_numpy()
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        return labels
