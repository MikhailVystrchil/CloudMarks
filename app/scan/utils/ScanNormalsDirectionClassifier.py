from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans

from CONFIG import RANDOM_SEED

if TYPE_CHECKING:
    from app.scan.Scan import Scan


class ScanNormalsDirectionClassifier:
    """
    Классифицирует точки Scan по направлениям локальных нормалей.
    """

    def __init__(
        self,
        scan: Scan,
    ) -> None:
        self.scan = scan

    def _normals_to_numpy(self) -> np.ndarray:
        normals: list[np.ndarray] = []

        for point in self.scan:
            normal = point.normals

            if normal is None:
                raise AttributeError(
                    "У точки отсутствует normals. "
                    "Сначала вызовите scan.compute_normals()."
                )

            normal_array = np.asarray(
                normal,
                dtype=np.float64,
            )

            if normal_array.shape != (3,):
                raise ValueError(
                    "Нормаль каждой точки должна иметь форму (3,)."
                )

            normals.append(normal_array)

        if not normals:
            raise ValueError(
                "Невозможно классифицировать нормали пустого Scan."
            )

        normal_array = np.asarray(
            normals,
            dtype=np.float64,
        )

        normal_lengths = np.linalg.norm(
            normal_array,
            axis=1,
        )

        if np.any(normal_lengths <= 1e-15):
            raise ValueError(
                "Обнаружена нулевая нормаль точки."
            )

        return normal_array / normal_lengths[:, np.newaxis]

    @staticmethod
    def _unify_normals_hemisphere(
        normals: np.ndarray,
    ) -> np.ndarray:
        """
        Ориентирует нормали в одно полупространство перед KMeans.

        Направления ``n`` и ``-n`` задают одну плоскость, поэтому без
        унификации KMeans может искусственно разделить одну поверхность.
        """
        normalized = normals.copy()

        reference = normalized.mean(axis=0)
        reference_norm = float(np.linalg.norm(reference))

        if reference_norm <= 1e-15:
            reference = np.asarray(
                [0.0, 0.0, 1.0],
                dtype=np.float64,
            )
        else:
            reference = reference / reference_norm

        negative_mask = normalized @ reference < 0.0
        normalized[negative_mask] *= -1.0

        return normalized

    def classify_normals(
        self,
        *,
        n_classes: int = 3,
        unify_hemisphere: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Выполняет KMeans-классификацию направлений нормалей.

        Возвращает:
        - labels: метки каждой точки;
        - centers: центры классов в пространстве направлений.
        """
        if n_classes < 1:
            raise ValueError(
                "n_classes должен быть положительным."
            )

        if len(self.scan) < n_classes:
            raise ValueError(
                "Количество точек Scan должно быть не меньше n_classes."
            )

        normals = self._normals_to_numpy()

        if unify_hemisphere:
            normals = self._unify_normals_hemisphere(normals)

        kmeans = KMeans(
            n_clusters=n_classes,
            n_init=10,
            random_state=RANDOM_SEED,
        )
        labels = kmeans.fit_predict(normals)

        for point, label in zip(self.scan, labels):
            point.labels = int(label)

        return labels, kmeans.cluster_centers_
