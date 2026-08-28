from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from loguru import logger
from sklearn.cluster import DBSCAN

from app.cross_points.CrossPoint import CrossPoint
from app.scan.Scan import Scan
from app.scan.ScanPlane import ScanPlane
from app.scan.ScanPoint import ScanPoint
from app.scan.plane_fitters.IterativePlaneFitter import IterativePlaneFitter
from app.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.scan.utils.ScanNormalsDirectionClassifier import (
    ScanNormalsDirectionClassifier,
)
from app.scan.utils.ScanSplitterByLabels import ScanSplitterByLabels


COND_THRESHOLD = 1_000.0
PARALLEL_ANGLE_TOL = np.deg2rad(10.0)

DEFAULT_NORMAL_K = 12
DEFAULT_MIN_POINTS_PER_PLANE = 15
DEFAULT_CLUSTER_EPS = 0.08
DEFAULT_CLUSTER_MIN_SAMPLES = 3


class PlaneGeometryStatus:
    GOOD = "GOOD"
    PARALLEL = "PARALLEL"
    ILL_CONDITIONED = "ILL_CONDITIONED"
    SINGULAR = "SINGULAR"


class PlaneGeometryDiagnostics:
    """
    Контроль геометрической устойчивости пересечения трёх плоскостей.
    """

    def __init__(
        self,
        planes: Sequence[ScanPlane],
        cond_threshold: float = COND_THRESHOLD,
        angle_tol_rad: float = PARALLEL_ANGLE_TOL,
    ) -> None:
        if len(planes) != 3:
            raise ValueError(
                "Для диагностики необходимы ровно три плоскости."
            )

        self.cond_threshold = float(cond_threshold)
        self.angle_tol_rad = float(angle_tol_rad)

        self.N = np.asarray(
            [[plane.A, plane.B, plane.C] for plane in planes],
            dtype=np.float64,
        )
        self.det = float(np.linalg.det(self.N))

        _, self.singular_values, _ = np.linalg.svd(self.N)
        smallest = float(self.singular_values[-1])
        self.cond = (
            float(self.singular_values[0] / smallest)
            if smallest > 1e-15
            else float("inf")
        )

        self.has_parallel = self._check_parallel(planes)
        self.messages: list[str] = []
        self.status = self._evaluate()

    def _check_parallel(
        self,
        planes: Sequence[ScanPlane],
    ) -> bool:
        normals = np.asarray(
            [plane.normal for plane in planes],
            dtype=np.float64,
        )

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms > 1e-15, norms, 1.0)

        cosine_tolerance = np.cos(self.angle_tol_rad)

        for i in range(3):
            for j in range(i + 1, 3):
                cosine = abs(float(np.dot(normals[i], normals[j])))
                angle_degrees = float(
                    np.rad2deg(
                        np.arccos(np.clip(cosine, -1.0, 1.0))
                    )
                )

                logger.debug(
                    "Угол между плоскостями {} и {}: {:.3f}°",
                    i + 1,
                    j + 1,
                    angle_degrees,
                )

                if cosine >= cosine_tolerance:
                    return True

        return False

    def _evaluate(self) -> str:
        if abs(self.det) < 1e-10:
            self.messages.append(
                f"det(N)={self.det:.3e}: матрица нормалей вырождена"
            )
            return PlaneGeometryStatus.SINGULAR

        if self.has_parallel:
            self.messages.append(
                "Обнаружены почти параллельные плоскости "
                f"(допуск {np.rad2deg(self.angle_tol_rad):.1f}°)"
            )
            return PlaneGeometryStatus.PARALLEL

        if self.cond > self.cond_threshold:
            self.messages.append(
                f"cond(N)={self.cond:.1f} > "
                f"{self.cond_threshold:.1f}: "
                "геометрия плохо обусловлена"
            )
            return PlaneGeometryStatus.ILL_CONDITIONED

        self.messages.append(
            f"det(N)={self.det:.6f}, cond(N)={self.cond:.2f}: "
            "геометрия устойчива"
        )
        return PlaneGeometryStatus.GOOD

    @property
    def is_reliable(self) -> bool:
        return self.status == PlaneGeometryStatus.GOOD

    def __str__(self) -> str:
        rows = [
            "PlaneGeometryDiagnostics:",
            f"  status          = {self.status}",
            f"  det(N)          = {self.det:.6f}",
            f"  cond(N)         = {self.cond:.2f}",
            f"  singular_values = {self.singular_values}",
            f"  has_parallel    = {self.has_parallel}",
        ]
        rows.extend(
            f"  [!] {message}"
            for message in self.messages
        )
        return "\n".join(rows)


class CrossPointExacter:
    """
    Извлекает виртуальную точку пересечения трёх локальных плоскостей.

    В отличие от прежней схемы, каждая плоскость выбирается не по всему
    классу нормалей, а по пространственной компоненте данного класса,
    ближайшей к reference_xyz. Это связывает решение с конкретным
    конструктивным узлом и стабилизирует split-half эксперимент.
    """

    def __init__(
        self,
        file_path: str,
        reference_xyz: Sequence[float] | None = None,
        show_scans: bool = False,
        normal_k: int = DEFAULT_NORMAL_K,
        min_points_per_plane: int = DEFAULT_MIN_POINTS_PER_PLANE,
        cluster_eps: float = DEFAULT_CLUSTER_EPS,
        cluster_min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
    ) -> None:
        scan_name = os.path.basename(file_path).rsplit(".", 1)[0]
        scan = Scan(scan_name)
        scan.import_points_from_file(
            file_path,
            compute_normals=False,
        )

        self._initialize(
            scan=scan,
            reference_xyz=reference_xyz,
            show_scans=show_scans,
            normal_k=normal_k,
            min_points_per_plane=min_points_per_plane,
            cluster_eps=cluster_eps,
            cluster_min_samples=cluster_min_samples,
        )

    @classmethod
    def from_scan(
        cls,
        scan: Scan,
        reference_xyz: Sequence[float] | None = None,
        show_scans: bool = False,
        normal_k: int = DEFAULT_NORMAL_K,
        min_points_per_plane: int = DEFAULT_MIN_POINTS_PER_PLANE,
        cluster_eps: float = DEFAULT_CLUSTER_EPS,
        cluster_min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
    ) -> "CrossPointExacter":
        """
        Создаёт извлекатель напрямую из локального Scan без временных файлов.
        """
        instance = cls.__new__(cls)

        instance._initialize(
            scan=scan,
            reference_xyz=reference_xyz,
            show_scans=show_scans,
            normal_k=normal_k,
            min_points_per_plane=min_points_per_plane,
            cluster_eps=cluster_eps,
            cluster_min_samples=cluster_min_samples,
        )

        return instance

    def _initialize(
        self,
        scan: Scan,
        reference_xyz: Sequence[float] | None,
        show_scans: bool,
        normal_k: int,
        min_points_per_plane: int,
        cluster_eps: float,
        cluster_min_samples: int,
    ) -> None:
        if len(scan) < 3 * min_points_per_plane:
            raise ValueError(
                f"В локальном фрагменте '{scan.name}' всего {len(scan)} точек; "
                f"необходимо хотя бы {3 * min_points_per_plane}."
            )

        if normal_k < 3:
            raise ValueError("normal_k должен быть не меньше 3.")

        if cluster_eps <= 0:
            raise ValueError("cluster_eps должен быть положительным.")

        if cluster_min_samples < 1:
            raise ValueError(
                "cluster_min_samples должен быть положительным."
            )

        self.base_scan = scan
        self.reference_xyz = self._validate_reference_xyz(reference_xyz)
        self.show_scans = bool(show_scans)

        self.normal_k = int(normal_k)
        self.min_points_per_plane = int(min_points_per_plane)
        self.cluster_eps = float(cluster_eps)
        self.cluster_min_samples = int(cluster_min_samples)

        self.planes: list[ScanPlane] | None = None
        self.cross_point: CrossPoint | None = None
        self.geometry_diagnostics: PlaneGeometryDiagnostics | None = None

        logger.info(
            "Подготовка виртуальной точки: scan='{}', N={}, "
            "reference={}",
            scan.name,
            len(scan),
            self.reference_xyz,
        )

        self.plane_scans = self._segment_local_planes()

    @staticmethod
    def _validate_reference_xyz(
        reference_xyz: Sequence[float] | None,
    ) -> np.ndarray | None:
        if reference_xyz is None:
            return None

        reference = np.asarray(reference_xyz, dtype=np.float64)

        if reference.shape != (3,):
            raise ValueError(
                "reference_xyz должен содержать координаты (X, Y, Z)."
            )

        if not np.all(np.isfinite(reference)):
            raise ValueError(
                "reference_xyz не должен содержать NaN или Inf."
            )

        return reference

    def _segment_local_planes(self) -> list[Scan]:
        """
        Делит локальный фрагмент на три направления нормалей, затем
        в каждом направлении выбирает пространственную компоненту,
        наиболее близкую к опорной точке.
        """
        logger.info(
            "Вычисление нормалей: k={}",
            self.normal_k,
        )
        self.base_scan.compute_normals(k=self.normal_k)

        logger.info(
            "Классификация направлений нормалей: n_classes=3"
        )
        classifier = ScanNormalsDirectionClassifier(self.base_scan)
        labels, _ = classifier.classify_normals(
            n_classes=3,
            unify_hemisphere=True,
        )

        unique_labels, counts = np.unique(
            labels,
            return_counts=True,
        )

        logger.info(
            "Размеры классов нормалей: {}",
            {
                int(label): int(count)
                for label, count in zip(unique_labels, counts)
            },
        )

        scans_by_normal = ScanSplitterByLabels(
            self.base_scan
        ).split()

        if len(scans_by_normal) != 3:
            raise ValueError(
                "Классификация нормалей не сформировала ровно три класса."
            )

        selected_scans: list[Scan] = []

        for plane_index, normal_label in enumerate(
            sorted(scans_by_normal),
            start=1,
        ):
            normal_scan = scans_by_normal[normal_label]

            selected_scan = self._select_component(
                normal_scan=normal_scan,
                plane_index=plane_index,
            )

            if len(selected_scan) < self.min_points_per_plane:
                raise ValueError(
                    f"Для плоскости {plane_index} выделено "
                    f"недостаточно точек: {len(selected_scan)}. "
                    f"Требуется не менее {self.min_points_per_plane}."
                )

            logger.info(
                "Плоскость {}/3: класс нормалей {}, "
                "выбрано {} точек",
                plane_index,
                normal_label,
                len(selected_scan),
            )

            if self.show_scans:
                selected_scan.plot()

            selected_scans.append(selected_scan)

        return selected_scans

    def _select_component(
        self,
        normal_scan: Scan,
        plane_index: int,
    ) -> Scan:
        """
        Выбирает компоненту класса нормалей.

        При наличии reference_xyz:
        - DBSCAN делит класс на пространственные компоненты;
        - выбирается компонента с минимальным расстоянием до опорной точки;
        - если DBSCAN не сформировал компоненту достаточного размера,
          выбираются ближайшие к опорной точке точки исходного класса.

        При отсутствии reference_xyz:
        - выбирается крупнейшая пространственная компонента;
        - при неудаче DBSCAN используется весь класс нормалей.
        """
        xyz = np.asarray(
            [
                [point.x, point.y, point.z]
                for point in normal_scan
            ],
            dtype=np.float64,
        )

        if len(xyz) < self.min_points_per_plane:
            return normal_scan

        dbscan_labels = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        ).fit_predict(xyz)

        unique_labels = np.unique(dbscan_labels)

        candidate_labels = [
            int(label)
            for label in unique_labels
            if label != -1
            and int(np.count_nonzero(dbscan_labels == label))
            >= self.min_points_per_plane
        ]

        if not candidate_labels:
            logger.debug(
                "Плоскость {}: DBSCAN не выделил компоненту "
                "размера >= {}; используется fallback.",
                plane_index,
                self.min_points_per_plane,
            )
            return self._fallback_component(
                normal_scan=normal_scan,
                xyz=xyz,
                plane_index=plane_index,
            )

        if self.reference_xyz is None:
            selected_label = max(
                candidate_labels,
                key=lambda label: int(
                    np.count_nonzero(dbscan_labels == label)
                ),
            )
        else:
            selected_label = min(
                candidate_labels,
                key=lambda label: float(
                    np.min(
                        np.linalg.norm(
                            xyz[dbscan_labels == label]
                            - self.reference_xyz,
                            axis=1,
                        )
                    )
                ),
            )

        selected_indices = np.flatnonzero(
            dbscan_labels == selected_label
        )

        logger.debug(
            "Плоскость {}: DBSCAN-компонента {}, "
            "точек={}",
            plane_index,
            selected_label,
            len(selected_indices),
        )

        return self._build_scan_from_indices(
            source_scan=normal_scan,
            indices=selected_indices,
            suffix=f"plane_{plane_index}_cluster_{selected_label}",
        )

    def _fallback_component(
        self,
        normal_scan: Scan,
        xyz: np.ndarray,
        plane_index: int,
    ) -> Scan:
        """
        Безопасный fallback.

        Для опорной точки выбирается ближайшая часть класса нормалей,
        но никогда не выбирается меньше min_points_per_plane.
        """
        if self.reference_xyz is None:
            logger.debug(
                "Плоскость {}: выбран весь класс нормалей "
                "без reference_xyz.",
                plane_index,
            )
            return normal_scan

        distances = np.linalg.norm(
            xyz - self.reference_xyz,
            axis=1,
        )

        selected_count = min(
            len(normal_scan),
            max(
                self.min_points_per_plane,
                int(np.ceil(len(normal_scan) * 0.25)),
            ),
        )

        selected_indices = np.argsort(distances)[:selected_count]

        logger.debug(
            "Плоскость {}: fallback ближайших точек, "
            "выбрано {} из {}.",
            plane_index,
            selected_count,
            len(normal_scan),
        )

        return self._build_scan_from_indices(
            source_scan=normal_scan,
            indices=selected_indices,
            suffix=f"plane_{plane_index}_fallback",
        )

    @staticmethod
    def _build_scan_from_indices(
        source_scan: Scan,
        indices: np.ndarray,
        suffix: str,
    ) -> Scan:
        """
        Создаёт независимый Scan, чтобы последующие операции очистки
        не меняли исходное локальное облако.
        """
        selected_scan = Scan(
            scan_name=f"{source_scan.name}__{suffix}"
        )

        source_points = list(source_scan)

        for index in np.asarray(indices, dtype=np.intp):
            point = source_points[int(index)]

            selected_scan.add_point(
                ScanPoint(
                    x=float(point.x),
                    y=float(point.y),
                    z=float(point.z),
                    color=getattr(point, "color", (0, 0, 0)),
                    normals=getattr(point, "normals", None),
                )
            )

        return selected_scan

    def calculate_planes(
        self,
        base_fitter=PlaneL1Fitter,
        mse_threshold: float = 0.0001,
        max_iteration: int = 20,
        k_sigma: float = 2.0,
    ) -> list[ScanPlane]:
        """
        Робастно очищает точки каждой плоскости и строит окончательные
        МНК-плоскости.
        """
        scan_planes: list[ScanPlane] = []

        for index, scan in enumerate(self.plane_scans, start=1):
            logger.info(
                "Аппроксимация плоскости {}/3: {} точек",
                index,
                len(scan),
            )

            scan_plane = ScanPlane.fit_plane_to_scan(
                scan=scan,
                fitter=IterativePlaneFitter,
                base_fitter=base_fitter,
                mse_threshold=mse_threshold,
                max_iteration=max_iteration,
                k_sigma=k_sigma,
                min_points=self.min_points_per_plane,
            )

            if len(scan_plane.scan) < self.min_points_per_plane:
                raise ValueError(
                    f"После очистки в плоскости {index} осталось "
                    f"{len(scan_plane.scan)} точек; требуется не менее "
                    f"{self.min_points_per_plane}."
                )

            logger.success(
                "Плоскость {}/3: inliers={}, RMSE={:.6f}",
                index,
                len(scan_plane.scan),
                scan_plane.mse,
            )

            scan_planes.append(scan_plane)

        self.planes = scan_planes
        return scan_planes

    def diagnose_geometry(
        self,
        cond_threshold: float = COND_THRESHOLD,
        angle_tol_rad: float = PARALLEL_ANGLE_TOL,
    ) -> PlaneGeometryDiagnostics:
        if self.planes is None:
            raise RuntimeError(
                "Сначала необходимо вычислить плоскости."
            )

        diagnostics = PlaneGeometryDiagnostics(
            planes=self.planes,
            cond_threshold=cond_threshold,
            angle_tol_rad=angle_tol_rad,
        )

        self.geometry_diagnostics = diagnostics

        if diagnostics.is_reliable:
            logger.success(
                "Геометрия устойчива: det(N)={:.6f}, "
                "cond(N)={:.2f}",
                diagnostics.det,
                diagnostics.cond,
            )
        else:
            logger.warning(
                "Геометрия неустойчива: status={}, {}",
                diagnostics.status,
                "; ".join(diagnostics.messages),
            )

        return diagnostics

    @staticmethod
    def _fallback_cov_from_mse(
        plane: ScanPlane,
    ) -> np.ndarray:
        sigma_squared = float(plane.mse ** 2)
        return np.eye(4, dtype=np.float64) * sigma_squared

    @staticmethod
    def _propagate_covariance(
        planes: Sequence[ScanPlane],
    ) -> np.ndarray:
        """
        Переносит ковариации параметров трёх плоскостей на координаты
        пересечения: K_X = J K_p J^T.
        """
        normal_matrix = np.asarray(
            [
                [plane.A, plane.B, plane.C]
                for plane in planes
            ],
            dtype=np.float64,
        )

        d_vector = np.asarray(
            [plane.D for plane in planes],
            dtype=np.float64,
        )

        xyz = np.linalg.solve(normal_matrix, -d_vector)
        x_coord, y_coord, z_coord = xyz

        inverse_normal_matrix = np.linalg.inv(normal_matrix)

        covariance_parameters = np.zeros(
            (12, 12),
            dtype=np.float64,
        )

        for index, plane in enumerate(planes):
            covariance = getattr(
                plane,
                "cov_params",
                None,
            )

            if covariance is None:
                covariance = (
                    CrossPointExacter._fallback_cov_from_mse(plane)
                )

            start = 4 * index
            covariance_parameters[
                start:start + 4,
                start:start + 4,
            ] = np.asarray(covariance, dtype=np.float64)

        jacobian = np.zeros((3, 12), dtype=np.float64)

        for index in range(3):
            column = inverse_normal_matrix[:, index]
            start = 4 * index

            jacobian[:, start] = -x_coord * column
            jacobian[:, start + 1] = -y_coord * column
            jacobian[:, start + 2] = -z_coord * column
            jacobian[:, start + 3] = -column

        covariance_xyz = (
            jacobian
            @ covariance_parameters
            @ jacobian.T
        )

        return 0.5 * (
            covariance_xyz
            + covariance_xyz.T
        )

    def calculate_intersect_point(
        self,
        cond_threshold: float = COND_THRESHOLD,
        angle_tol_rad: float = PARALLEL_ANGLE_TOL,
    ) -> CrossPoint:
        """
        Находит точку пересечения трёх плоскостей и оценивает ковариацию.
        """
        if self.planes is None:
            raise RuntimeError(
                "Сначала необходимо вычислить плоскости."
            )

        diagnostics = self.diagnose_geometry(
            cond_threshold=cond_threshold,
            angle_tol_rad=angle_tol_rad,
        )

        if diagnostics.status == PlaneGeometryStatus.SINGULAR:
            raise ValueError(
                "Плоскости не имеют единственной точки пересечения: "
                f"{diagnostics.messages[0]}"
            )

        normal_matrix = np.asarray(
            [
                [plane.A, plane.B, plane.C]
                for plane in self.planes
            ],
            dtype=np.float64,
        )

        right_side = np.asarray(
            [-plane.D for plane in self.planes],
            dtype=np.float64,
        )

        xyz = np.linalg.solve(normal_matrix, right_side)

        self.cross_point = CrossPoint(
            name=self.base_scan.name,
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]),
        )

        self.cross_point.status = diagnostics.status
        self.cross_point.load_mses(
            plane_mses=[
                plane.mse
                for plane in self.planes
            ]
        )

        if diagnostics.is_reliable:
            covariance_xyz = self._propagate_covariance(
                self.planes
            )
            self.cross_point.load_covariance(covariance_xyz)
        else:
            self.cross_point.mark_unreliable_accuracy()

        return self.cross_point

    def get_result_str(self) -> str:
        if self.cross_point is None:
            raise RuntimeError(
                "Точка пересечения ещё не вычислена."
            )

        point = self.cross_point

        values = [
            f"name={point.name}",
            f"x={point.x:.6f}",
            f"y={point.y:.6f}",
            f"z={point.z:.6f}",
            f"status={point.status}",
        ]

        if (
            point.reliable_accuracy
            and point.sigma_xyz is not None
        ):
            values.extend([
                f"sigma_x={point.sigma_xyz[0]:.6f}",
                f"sigma_y={point.sigma_xyz[1]:.6f}",
                f"sigma_z={point.sigma_xyz[2]:.6f}",
            ])
        else:
            values.append("accuracy=UNRELIABLE")

        return ", ".join(values)
