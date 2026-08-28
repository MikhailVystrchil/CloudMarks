import numpy as np
from sklearn.neighbors import NearestNeighbors


class ScanNormalsCalculator:

    def __init__(self, scan):
        self.scan = scan

    def compute_normals(self, *args, k=20, **kwargs):
        pts = self.__scan_to_numpy()
        normals = self.__compute_normals(pts, k=k)
        for p, n in zip(self.scan, normals):
            setattr(p, "normals", n)
        return normals

    def __scan_to_numpy(self):
        pts = np.array([[p.x, p.y, p.z] for p in self.scan])
        return pts

    @staticmethod
    def __compute_normals(points_xyz, k):
        N = points_xyz.shape[0]
        normals = np.zeros((N, 3), dtype=np.float64)

        nn = NearestNeighbors(n_neighbors=min(k, N), algorithm='kd_tree')
        nn.fit(points_xyz)
        distances, indices = nn.kneighbors(points_xyz)

        for i in range(N):
            neigh_idx = indices[i]
            neigh_pts = points_xyz[neigh_idx]

            centroid = neigh_pts.mean(axis=0)
            centered = neigh_pts - centroid

            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            n = vh[-1, :]

            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                n = n / n_norm

            normals[i] = n

        return normals
