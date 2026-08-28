import os

import numpy as np
from loguru import logger

from app.scan.Scan import Scan
from app.scan.ScanPlane import ScanPlane
from app.scan.plane_fitters.IterativePlaneFitter import IterativePlaneFitter
from app.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.scan.utils.ScanNormalsDirectionClassifier import ScanNormalsDirectionClassifier
from app.scan.utils.ScanSplitterByLabels import ScanSplitterByLabels
from app.cross_points.CrossPoint import CrossPoint


COND_THRESHOLD = 1_000.0
PARALLEL_ANGLE_TOL = np.deg2rad(10.0)


class PlaneGeometryStatus:
    GOOD = "GOOD"
    PARALLEL = "PARALLEL"
    ILL_CONDITIONED = "ILL_CONDITIONED"
    SINGULAR = "SINGULAR"


class PlaneGeometryDiagnostics:
    """
    Результат проверки устойчивости пересечения трёх плоскостей.

    Проверяются:
    - вырожденность матрицы нормалей;
    - наличие почти параллельных плоскостей;
    - число обусловленности матрицы нормалей.
    """

    def __init__(
        self,
        planes,
        cond_threshold: float = COND_THRESHOLD,
        angle_tol_rad: float = PARALLEL_ANGLE_TOL,
    ):
        if planes is None or len(planes) != 3:
            raise ValueError(
                "Для диагностики геометрии необходимы ровно три плоскости."
            )

        self.cond_threshold = cond_threshold
        self.angle_tol_rad = angle_tol_rad

        self.N = np.array(
            [[plane.A, plane.B, plane.C] for plane in planes],
            dtype=float,
        )
        self.det = float(np.linalg.det(self.N))

        _, singular_values, _ = np.linalg.svd(self.N)
        self.singular_values = singular_values
        self.cond = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 1e-15
            else float("inf")
        )

        self.has_parallel = self._check_parallel(planes)
        self.messages: list[str] = []
        self.status = self._evaluate()

    def _check_parallel(self, planes) -> bool:
        normals = np.array(
            [plane.normal for plane in planes],
            dtype=float,
        )
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms > 1e-15, norms, 1.0)

        cos_tolerance = np.cos(self.angle_tol_rad)

        for i in range(len(normals)):
            for j in range(i + 1, len(normals)):
                cosine = abs(float(np.dot(normals[i], normals[j])))
                angle_deg = np.rad2deg(
                    np.arccos(np.clip(cosine, -1.0, 1.0))
                )

                logger.debug(
                    "Угол между нормалями плоскостей {} и {}: {:.3f}°",
                    i + 1,
                    j + 1,
                    angle_deg,
                )

                if cosine >= cos_tolerance:
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
                f"cond(N)={self.cond:.1f} > {self.cond_threshold:.0f}: "
                "геометрия плохо обусловлена"
            )
            return PlaneGeometryStatus.ILL_CONDITIONED

        self.messages.append(
            f"det(N)={self.det:.4f}, cond(N)={self.cond:.1f}: "
            "геометрия устойчива"
        )
        return PlaneGeometryStatus.GOOD

    @property
    def is_reliable(self) -> bool:
        return self.status == PlaneGeometryStatus.GOOD

    def __str__(self) -> str:
        lines = [
            "PlaneGeometryDiagnostics:",
            f"  status          = {self.status}",
            f"  det(N)          = {self.det:.6f}",
            f"  cond(N)         = {self.cond:.2f}",
            f"  singular_values = {self.singular_values}",
            f"  has_parallel    = {self.has_parallel}",
        ]
        lines.extend(f"  [!] {message}" for message in self.messages)
        return "\n".join(lines)


class CrossPointExacter:
    """
    Вычисляет виртуальную точку как пересечение трёх плоскостей,
    выделенных из локального облака точек.

    Последовательность:
    1. Загрузка локального облака;
    2. Вычисление нормалей;
    3. Классификация нормалей KMeans на три направления;
    4. Разделение облака на три подскана;
    5. Робастная очистка выбросов и МНК-аппроксимация плоскостей;
    6. Контроль геометрии пересечения;
    7. Решение СЛАУ и перенос ковариаций на виртуальную точку.

    DBSCAN намеренно не используется: пространственную очистку выбросов
    выполняет IterativePlaneFitter, что не исключает малые, но геометрически
    значимые локальные фрагменты плоскостей.
    """

    def __init__(
        self,
        file_path: str,
        show_scans: bool = False,
    ):
        self.file_path = file_path

        logger.info("Инициализация извлекателя виртуальной точки")
        logger.info("Файл локального облака: {}", file_path)

        self.base_scan = self.__init_scan(file_path)
        self.plane_scans = self.__separate_plane_scans(
            show_scans=show_scans,
        )

        self.planes: list[ScanPlane] | None = None
        self.cross_point: CrossPoint | None = None
        self.geometry_diagnostics: PlaneGeometryDiagnostics | None = None

    @staticmethod
    def __init_scan(file_path: str) -> Scan:
        scan_name = os.path.basename(file_path).rsplit(".", maxsplit=1)[0]
        scan = Scan(scan_name)

        logger.info("Загрузка исходного облака точек")
        scan.import_points_from_file(file_path, compute_normals=False)
        logger.info("Исходное облако загружено: {} точек", len(scan))

        logger.info("Вычисление локальных нормалей: k=8")
        scan.compute_normals(k=8)

        logger.info("Классификация нормалей на 3 направления")
        normals_classifier = ScanNormalsDirectionClassifier(scan)
        labels, _ = normals_classifier.classify_normals(
            n_classes=3,
            unify_hemisphere=True,
        )

        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = {
            int(label): int(count)
            for label, count in zip(unique_labels, counts)
        }

        logger.info(
            "Классификация направлений нормалей завершена: {}",
            label_counts,
        )

        return scan

    def __separate_plane_scans(
        self,
        show_scans: bool,
    ) -> list[Scan]:
        """
        Разделяет точки на три локальных подскана по классам нормалей.

        Пространственная фильтрация DBSCAN не выполняется. Все точки каждого
        класса передаются IterativePlaneFitter, который очищает выбросы по
        расстоянию до робастно оценённой плоскости.
        """
        normal_scans = ScanSplitterByLabels(self.base_scan).split()

        if len(normal_scans) != 3:
            raise ValueError(
                "После классификации нормалей должно быть выделено три класса. "
                f"Фактически выделено: {len(normal_scans)}."
            )

        normal_classes_sizes = {
            int(class_label): len(class_scan)
            for class_label, class_scan in normal_scans.items()
        }

        logger.info(
            "Выделено классов направлений нормалей: {}. Размеры: {}",
            len(normal_scans),
            normal_classes_sizes,
        )

        plane_scans: list[Scan] = []

        for index, (normal_label, scan) in enumerate(
            normal_scans.items(),
            start=1,
        ):
            if len(scan) < 6:
                raise ValueError(
                    f"Для плоскости {index} выделено недостаточно точек: "
                    f"{len(scan)}. Необходимо не менее 6."
                )

            logger.info(
                "Плоскость {}/3: класс нормалей {}, {} точек. "
                "DBSCAN не применяется.",
                index,
                normal_label,
                len(scan),
            )

            if show_scans:
                scan.plot()

            plane_scans.append(scan)

        logger.success(
            "Локальные фрагменты трёх плоскостей подготовлены: {}",
            [len(scan) for scan in plane_scans],
        )

        return plane_scans

    def calculate_planes(
        self,
        base_fitter=PlaneL1Fitter,
        mse_threshold: float = 0.0001,
        max_iteration: int = 20,
        k_sigma: float = 2,
    ) -> list[ScanPlane]:
        logger.info(
            "Аппроксимация трёх плоскостей: mse_threshold={}, "
            "max_iteration={}, k_sigma={}",
            mse_threshold,
            max_iteration,
            k_sigma,
        )

        scan_planes: list[ScanPlane] = []

        for index, scan in enumerate(self.plane_scans, start=1):
            logger.info(
                "Построение плоскости {}/3 по {} точкам",
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
            )

            scan_planes.append(scan_plane)

            logger.success(
                "Плоскость {}/3 построена: inliers={}, RMSE={:.6f}, "
                "A={:.6f}, B={:.6f}, C={:.6f}, D={:.6f}",
                index,
                len(scan_plane.scan),
                scan_plane.mse,
                scan_plane.A,
                scan_plane.B,
                scan_plane.C,
                scan_plane.D,
            )

        self.planes = scan_planes
        return scan_planes

    def diagnose_geometry(
        self,
        cond_threshold: float = COND_THRESHOLD,
        angle_tol_rad: float = PARALLEL_ANGLE_TOL,
    ) -> PlaneGeometryDiagnostics:
        if self.planes is None:
            raise RuntimeError(
                "Плоскости ещё не вычислены. Сначала вызовите calculate_planes()."
            )

        logger.info(
            "Диагностика геометрии: cond_threshold={}, "
            "минимальный угол={:.2f}°",
            cond_threshold,
            np.rad2deg(angle_tol_rad),
        )

        diagnostics = PlaneGeometryDiagnostics(
            self.planes,
            cond_threshold=cond_threshold,
            angle_tol_rad=angle_tol_rad,
        )
        self.geometry_diagnostics = diagnostics

        if diagnostics.is_reliable:
            logger.success(
                "Геометрия устойчива: det(N)={:.6f}, cond(N)={:.2f}",
                diagnostics.det,
                diagnostics.cond,
            )
        else:
            logger.warning(
                "Геометрия неустойчива: status={}, det(N)={:.6f}, "
                "cond(N)={:.2f}; {}",
                diagnostics.status,
                diagnostics.det,
                diagnostics.cond,
                "; ".join(diagnostics.messages),
            )

        return diagnostics

    @staticmethod
    def _fallback_cov_from_mse(plane: ScanPlane) -> np.ndarray:
        """
        Консервативная запасная оценка ковариации, используемая только
        в случае отсутствия ковариации параметров плоскости.
        """
        sigma2 = float(plane.mse ** 2)
        return np.eye(4, dtype=float) * sigma2

    @staticmethod
    def _propagate_covariance(planes: list[ScanPlane]) -> np.ndarray:
        """
        Перенос ковариаций параметров трёх плоскостей на координаты
        виртуальной точки по формуле первого порядка:

            K_X = J K_p J^T.

        В рамках локальной модели K_p принимается блочно-диагональной:
        параметры плоскостей рассматриваются как независимые.
        """
        normal_matrix = np.array(
            [[plane.A, plane.B, plane.C] for plane in planes],
            dtype=float,
        )
        d_vector = np.array(
            [plane.D for plane in planes],
            dtype=float,
        )

        xyz = np.linalg.solve(normal_matrix, -d_vector)
        x_coord, y_coord, z_coord = xyz
        normal_matrix_inv = np.linalg.inv(normal_matrix)

        covariance_params = np.zeros((12, 12), dtype=float)

        for plane_index, plane in enumerate(planes):
            covariance = (
                plane.cov_params
                if getattr(plane, "cov_params", None) is not None
                else CrossPointExacter._fallback_cov_from_mse(plane)
            )

            start = 4 * plane_index
            covariance_params[start:start + 4, start:start + 4] = covariance

        jacobian = np.zeros((3, 12), dtype=float)

        for plane_index in range(3):
            column = normal_matrix_inv[:, plane_index]
            start = 4 * plane_index

            jacobian[:, start] = -x_coord * column
            jacobian[:, start + 1] = -y_coord * column
            jacobian[:, start + 2] = -z_coord * column
            jacobian[:, start + 3] = -column

        covariance_xyz = jacobian @ covariance_params @ jacobian.T
        covariance_xyz = 0.5 * (
            covariance_xyz + covariance_xyz.T
        )

        eigenvalues = np.linalg.eigvalsh(covariance_xyz)
        if np.any(eigenvalues < -1e-12):
            logger.warning(
                "Получена не положительно полуопределённая ковариация точки. "
                "Минимальное собственное значение: {:.3e}",
                float(eigenvalues.min()),
            )

        return covariance_xyz

    def calculate_intersect_point(
        self,
        cond_threshold: float = COND_THRESHOLD,
        angle_tol_rad: float = PARALLEL_ANGLE_TOL,
    ) -> CrossPoint:
        if self.planes is None:
            raise RuntimeError(
                "Плоскости ещё не вычислены. Сначала вызовите calculate_planes()."
            )

        diagnostics = self.diagnose_geometry(
            cond_threshold=cond_threshold,
            angle_tol_rad=angle_tol_rad,
        )

        normal_matrix = np.array(
            [[plane.A, plane.B, plane.C] for plane in self.planes],
            dtype=float,
        )
        right_side = np.array(
            [-plane.D for plane in self.planes],
            dtype=float,
        )

        if diagnostics.status == PlaneGeometryStatus.SINGULAR:
            logger.error(
                "Плоскости не имеют единственной точки пересечения: {}",
                diagnostics.messages[0],
            )
            raise ValueError(
                "Плоскости не имеют единственной точки пересечения: "
                f"{diagnostics.messages[0]}"
            )

        logger.info("Решение СЛАУ для координат точки пересечения")
        xyz = np.linalg.solve(normal_matrix, right_side)

        self.cross_point = CrossPoint(
            name=self.base_scan.name,
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]),
        )
        self.cross_point.status = diagnostics.status

        plane_mses = [plane.mse for plane in self.planes]
        self.cross_point.load_mses(plane_mses=plane_mses)

        if diagnostics.is_reliable:
            logger.info(
                "Перенос ковариаций параметров плоскостей на виртуальную точку"
            )

            covariance_xyz = self._propagate_covariance(self.planes)
            self.cross_point.load_covariance(covariance_xyz)

            logger.success(
                "Ковариация точки определена: sigma_X={:.6f}, "
                "sigma_Y={:.6f}, sigma_Z={:.6f}",
                self.cross_point.sigma_xyz[0],
                self.cross_point.sigma_xyz[1],
                self.cross_point.sigma_xyz[2],
            )
        else:
            self.cross_point.mark_unreliable_accuracy()

            logger.warning(
                "Ковариация точки '{}' не вычислена: status={}, "
                "cond(N)={:.2f}",
                self.base_scan.name,
                diagnostics.status,
                diagnostics.cond,
            )

        return self.cross_point

    def get_result_str(self) -> str:
        if self.cross_point is None:
            raise RuntimeError(
                "Точка пересечения ещё не вычислена. "
                "Сначала вызовите calculate_intersect_point()."
            )

        point = self.cross_point

        parts = [
            self.base_scan.name,
            f"x={point.x:.6f}",
            f"y={point.y:.6f}",
            f"z={point.z:.6f}",
            f"status={point.status}",
        ]

        if point.reliable_accuracy and point.sigma_xyz is not None:
            sigma_x, sigma_y, sigma_z = point.sigma_xyz
            parts.extend([
                f"sigma_x={sigma_x:.6f}",
                f"sigma_y={sigma_y:.6f}",
                f"sigma_z={sigma_z:.6f}",
            ])
        else:
            parts.append("accuracy=UNRELIABLE")

        if self.geometry_diagnostics is not None:
            diagnostics = self.geometry_diagnostics
            parts.extend([
                f"cond(N)={diagnostics.cond:.2f}",
                f"det(N)={diagnostics.det:.6f}",
            ])

        return ", ".join(parts)
