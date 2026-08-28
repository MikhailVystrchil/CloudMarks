from app.scan.filters.ScanFilterABC import ScanFilterABC
from app.scan.utils.ScanDBSCANClusterizator import ScanDBSCANClusterizator


class ScanFilterByDBSCAN(ScanFilterABC):

    def __init__(self, eps=0.05, min_samples=100, min_cluster_size=None):
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size

    def __compute_clusters(self, scan):
        s_dbscan_c = ScanDBSCANClusterizator(scan)
        s_dbscan_c.compute_clusters(eps=self.eps, min_samples=self.min_samples)

    def filter(self, scan):
        self.__compute_clusters(scan)

        if self.min_cluster_size is None:
            return [point for point in scan if getattr(point, "labels", -1) != -1]

        cluster_counts = {}
        for point in scan:
            lbl = getattr(point, "labels", -1)
            if lbl == -1:
                continue
            cluster_counts[lbl] = cluster_counts.get(lbl, 0) + 1

        large_clusters = {
            lbl for lbl, cnt in cluster_counts.items()
            if cnt >= self.min_cluster_size
        }

        return [point for point in scan if getattr(point, "labels", -1) in large_clusters]
