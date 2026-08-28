from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from app.scan.Scan import Scan


class ScanNormalsCalculator:
    """
    Вычисляет локальные нормали точек облака методом PCA
    по ближайшим соседям.
    """

    def __init__(
        self,
        scan: Scan,
    ) -> None:
        self.scan = scan

    def compute_normals(
        self,
        *,
        k: int = 20,
    ) -> np.ndarray:
        """
        Вычисляет нормали и сохраняет их в ``ScanPoint.normals``.

        Parameters
        ----------
        k:
            Число ближайших соседей, используемых при локальном PCA.

        Returns
        -------
        np.ndarray
            Массив нормалей формы ``(N, 3)``.
        """
        if len(self.scan) < 3:
            raise ValueError(
                "Для вычисления нормалей необходимо не менее 3 точек."
            )

        if k < 3:
            raise ValueError(
                "k должен быть не меньше 3."
            )

        points_xyz = self.scan.to_numpy()

        normals = self._compute_normals(
            points_xyz=points_xyz,
            k=k,
        )

        for point, normal in zip(self.scan, normals):
            point.normals = normal

        return normals

    @staticmethod
    def _compute_normals(
        points_xyz: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Вычисляет единичную нормаль каждой точки через SVD
        её локальной окрестности.
        """
        point_count = points_xyz.shape[0]

        normals = np.zeros(
            shape=(point_count, 3),
            dtype=np.float64,
        )

        neighbors_count = min(k, point_count)

        nearest_neighbors = NearestNeighbors(
            n_neighbors=neighbors_count,
            algorithm="kd_tree",
        )
        nearest_neighbors.fit(points_xyz)

        _, neighbor_indices = nearest_neighbors.kneighbors(
            points_xyz
        )

        for point_index, indices in enumerate(
            neighbor_indices
        ):
            neighborhood = points_xyz[indices]

            centroid = neighborhood.mean(axis=0)
            centered = neighborhood - centroid

            _, _, right_singular_vectors = np.linalg.svd(
                centered,
                full_matrices=False,
            )

            normal = right_singular_vectors[-1]
            normal_norm = float(
                np.linalg.norm(normal)
            )

            if normal_norm > 1e-15:
                normal = normal / normal_norm

            normals[point_index] = normal

        return normals
