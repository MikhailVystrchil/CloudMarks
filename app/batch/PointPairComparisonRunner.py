from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from app.batch.ReferencePoint import ReferencePoint
from app.batch.ReferencePointReader import ReferencePointReader
from app.batch.ScanNeighborhoodExtractor import (
    ScanNeighborhoodExtractor,
)
from app.cross_points.CrossPointExacter import CrossPointExacter
from app.deformation.DeformationAnalyzer import DeformationAnalyzer
from app.scan.Scan import Scan


@dataclass
class ReferencePointProcessingResult:
    """
    Результат обработки одной опорной точки для двух эпох.

    processing_status:
        SUCCESS    — обе точки прошли все геометрические и точностные проверки;
        UNRELIABLE — координаты были вычислены, но не прошли контроль качества;
        FAILED     — не удалось извлечь обе виртуальные точки.

    Только SUCCESS попадает в DeformationAnalyzer.
    """

    name: str
    reference_x: float
    reference_y: float
    reference_z: float
    radius: float

    epoch1_points: int | None = None
    epoch2_points: int | None = None

    epoch1_reference_distance: float | None = None
    epoch2_reference_distance: float | None = None
    pair_distance: float | None = None

    processing_status: str = "FAILED"
    processing_message: str = ""

    point_epoch1: Any | None = None
    point_epoch2: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "reference_x": self.reference_x,
            "reference_y": self.reference_y,
            "reference_z": self.reference_z,
            "radius": self.radius,
            "epoch1_neighborhood_points": self.epoch1_points,
            "epoch2_neighborhood_points": self.epoch2_points,
            "epoch1_reference_distance": self.epoch1_reference_distance,
            "epoch2_reference_distance": self.epoch2_reference_distance,
            "pair_distance": self.pair_distance,
            "processing_status": self.processing_status,
            "processing_message": self.processing_message,
        }

        self._append_virtual_point(
            result=result,
            point=self.point_epoch1,
            prefix="epoch1",
        )
        self._append_virtual_point(
            result=result,
            point=self.point_epoch2,
            prefix="epoch2",
        )

        return result

    @staticmethod
    def _append_virtual_point(
        result: dict[str, Any],
        point: Any | None,
        prefix: str,
    ) -> None:
        if point is None:
            result.update({
                f"{prefix}_x": np.nan,
                f"{prefix}_y": np.nan,
                f"{prefix}_z": np.nan,
                f"{prefix}_status": None,
                f"{prefix}_reliable_accuracy": False,
                f"{prefix}_sigma_x": np.nan,
                f"{prefix}_sigma_y": np.nan,
                f"{prefix}_sigma_z": np.nan,
            })
            return

        result.update({
            f"{prefix}_x": float(point.x),
            f"{prefix}_y": float(point.y),
            f"{prefix}_z": float(point.z),
            f"{prefix}_status": str(point.status),
            f"{prefix}_reliable_accuracy": bool(
                point.reliable_accuracy
            ),
        })

        if point.sigma_xyz is None:
            result.update({
                f"{prefix}_sigma_x": np.nan,
                f"{prefix}_sigma_y": np.nan,
                f"{prefix}_sigma_z": np.nan,
            })
            return

        result.update({
            f"{prefix}_sigma_x": float(point.sigma_xyz[0]),
            f"{prefix}_sigma_y": float(point.sigma_xyz[1]),
            f"{prefix}_sigma_z": float(point.sigma_xyz[2]),
        })


class PointPairComparisonRunner:
    """
    Пакетно сравнивает два зарегистрированных облака точек.

    Для каждой опорной точки:
    1. Находит точки в локальной сферической окрестности двух сканов.
    2. Извлекает виртуальную точку пересечения трёх плоскостей в каждой эпохе.
    3. Отвергает ненадёжные, геометрически неверные и несопоставимые пары.
    4. Передаёт только пригодные пары в DeformationAnalyzer.

    Важно:
    Исходные сканы должны быть предварительно приведены к одной системе
    координат. Данный класс не выполняет взаимную регистрацию эпох.
    """

    SUCCESS = "SUCCESS"
    UNRELIABLE = "UNRELIABLE"
    FAILED = "FAILED"

    def __init__(
        self,
        scan_epoch1: Scan,
        scan_epoch2: Scan,
        default_radius: float,
        alpha: float = 0.05,
        min_neighborhood_points: int = 60,
        min_points_per_plane: int = 15,
        max_reference_distance_factor: float = 1.25,
        max_pair_distance: float | None = None,
        normal_k: int = 12,
        cluster_eps: float = 0.08,
        cluster_min_samples: int = 3,
    ) -> None:
        if default_radius <= 0:
            raise ValueError(
                "default_radius должен быть положительным."
            )

        if not 0.0 < alpha < 1.0:
            raise ValueError(
                "alpha должен принадлежать интервалу (0, 1)."
            )

        if min_points_per_plane < 6:
            raise ValueError(
                "min_points_per_plane должен быть не менее 6."
            )

        if min_neighborhood_points < 3 * min_points_per_plane:
            raise ValueError(
                "min_neighborhood_points должен быть не меньше "
                "3 * min_points_per_plane."
            )

        if max_reference_distance_factor <= 0:
            raise ValueError(
                "max_reference_distance_factor должен быть положительным."
            )

        if normal_k < 3:
            raise ValueError(
                "normal_k должен быть не менее 3."
            )

        if cluster_eps <= 0:
            raise ValueError(
                "cluster_eps должен быть положительным."
            )

        if cluster_min_samples < 1:
            raise ValueError(
                "cluster_min_samples должен быть положительным."
            )

        self.scan_epoch1 = scan_epoch1
        self.scan_epoch2 = scan_epoch2

        self.default_radius = float(default_radius)
        self.alpha = float(alpha)

        self.min_neighborhood_points = int(
            min_neighborhood_points
        )
        self.min_points_per_plane = int(
            min_points_per_plane
        )

        self.max_reference_distance_factor = float(
            max_reference_distance_factor
        )

        self.max_pair_distance = (
            float(max_pair_distance)
            if max_pair_distance is not None
            else self.default_radius * 0.05
        )

        if self.max_pair_distance <= 0:
            raise ValueError(
                "max_pair_distance должен быть положительным."
            )

        self.normal_k = int(normal_k)
        self.cluster_eps = float(cluster_eps)
        self.cluster_min_samples = int(cluster_min_samples)

        logger.info(
            "Создание cKDTree для epoch1: {} точек",
            len(scan_epoch1),
        )
        self.extractor_epoch1 = ScanNeighborhoodExtractor(
            scan=scan_epoch1
        )

        logger.info(
            "Создание cKDTree для epoch2: {} точек",
            len(scan_epoch2),
        )
        self.extractor_epoch2 = ScanNeighborhoodExtractor(
            scan=scan_epoch2
        )

        self.processing_results: list[
            ReferencePointProcessingResult
        ] = []
        self.deformation_results: list[Any] = []

    @classmethod
    def from_files(
        cls,
        epoch1_path: str | Path,
        epoch2_path: str | Path,
        default_radius: float,
        alpha: float = 0.05,
        min_neighborhood_points: int = 60,
        min_points_per_plane: int = 15,
        max_reference_distance_factor: float = 1.25,
        max_pair_distance: float | None = None,
        normal_k: int = 12,
        cluster_eps: float = 0.08,
        cluster_min_samples: int = 3,
    ) -> "PointPairComparisonRunner":
        """
        Загружает каждый большой скан только один раз.
        """
        scan_epoch1 = cls._load_scan(
            file_path=epoch1_path,
            scan_name="epoch1",
        )
        scan_epoch2 = cls._load_scan(
            file_path=epoch2_path,
            scan_name="epoch2",
        )

        return cls(
            scan_epoch1=scan_epoch1,
            scan_epoch2=scan_epoch2,
            default_radius=default_radius,
            alpha=alpha,
            min_neighborhood_points=min_neighborhood_points,
            min_points_per_plane=min_points_per_plane,
            max_reference_distance_factor=max_reference_distance_factor,
            max_pair_distance=max_pair_distance,
            normal_k=normal_k,
            cluster_eps=cluster_eps,
            cluster_min_samples=cluster_min_samples,
        )

    @staticmethod
    def _load_scan(
        file_path: str | Path,
        scan_name: str,
    ) -> Scan:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Файл скана не найден: {path}"
            )

        logger.info(
            "Загрузка большого скана '{}': {}",
            scan_name,
            path,
        )

        scan = Scan(scan_name)
        scan.import_points_from_file(
            file_path=str(path),
            compute_normals=False,
        )

        if len(scan) == 0:
            raise ValueError(
                f"Скан '{path}' не содержит точек."
            )

        logger.success(
            "Скан '{}' загружен: {} точек",
            scan_name,
            len(scan),
        )

        return scan

    def run_from_reference_file(
        self,
        reference_points_path: str | Path,
        workers: int = -1,
        fail_on_point_error: bool = False,
    ) -> pd.DataFrame:
        reference_points = ReferencePointReader.read(
            reference_points_path
        )

        return self.run(
            reference_points=reference_points,
            workers=workers,
            fail_on_point_error=fail_on_point_error,
        )

    def run(
        self,
        reference_points: list[ReferencePoint],
        workers: int = -1,
        fail_on_point_error: bool = False,
    ) -> pd.DataFrame:
        """
        Находит все окрестности массово, а локальные Scan формирует
        последовательно, сохраняя разумное потребление памяти.
        """
        if not reference_points:
            raise ValueError(
                "Список опорных точек пуст."
            )

        if workers == 0 or workers < -1:
            raise ValueError(
                "workers должен быть -1 или положительным целым числом."
            )

        point_names = [
            reference_point.name
            for reference_point in reference_points
        ]

        if len(point_names) != len(set(point_names)):
            raise ValueError(
                "Имена опорных точек должны быть уникальными."
            )

        centers = np.asarray(
            [
                reference_point.as_array()
                for reference_point in reference_points
            ],
            dtype=np.float64,
        )

        radii = np.asarray(
            [
                reference_point.radius
                if reference_point.radius is not None
                else self.default_radius
                for reference_point in reference_points
            ],
            dtype=np.float64,
        )

        if (
            not np.all(np.isfinite(radii))
            or np.any(radii <= 0)
        ):
            raise ValueError(
                "Все радиусы окрестностей должны быть положительными."
            )

        logger.info(
            "Старт обработки: опорных точек={}, workers={}, "
            "d_ref_factor={}, d_pair_max={} м",
            len(reference_points),
            workers,
            self.max_reference_distance_factor,
            self.max_pair_distance,
        )

        self.processing_results = []
        self.deformation_results = []

        logger.info(
            "Пакетный поиск окрестностей для epoch1"
        )
        epoch1_indices_sets = (
            self.extractor_epoch1.query_indices_many(
                centers=centers,
                radius=radii,
                workers=workers,
            )
        )

        logger.info(
            "Пакетный поиск окрестностей для epoch2"
        )
        epoch2_indices_sets = (
            self.extractor_epoch2.query_indices_many(
                centers=centers,
                radius=radii,
                workers=workers,
            )
        )

        valid_points_epoch1: list[Any] = []
        valid_points_epoch2: list[Any] = []

        for index, (
            reference_point,
            radius,
            epoch1_indices,
            epoch2_indices,
        ) in enumerate(
            zip(
                reference_points,
                radii,
                epoch1_indices_sets,
                epoch2_indices_sets,
            ),
            start=1,
        ):
            logger.info(
                "Точка {}/{}: '{}'",
                index,
                len(reference_points),
                reference_point.name,
            )

            neighborhood_epoch1 = (
                self.extractor_epoch1.extract_by_indices(
                    point_indices=epoch1_indices,
                    reference_name=reference_point.name,
                    radius=float(radius),
                )
            )

            neighborhood_epoch2 = (
                self.extractor_epoch2.extract_by_indices(
                    point_indices=epoch2_indices,
                    reference_name=reference_point.name,
                    radius=float(radius),
                )
            )

            result = self._process_reference_point(
                reference_point=reference_point,
                neighborhood_epoch1=neighborhood_epoch1,
                neighborhood_epoch2=neighborhood_epoch2,
            )

            self.processing_results.append(result)

            del neighborhood_epoch1
            del neighborhood_epoch2

            if result.processing_status == self.SUCCESS:
                valid_points_epoch1.append(result.point_epoch1)
                valid_points_epoch2.append(result.point_epoch2)

                logger.success(
                    "Точка '{}' принята для анализа",
                    result.name,
                )
                continue

            logger.warning(
                "Точка '{}' исключена: status={}, причина={}",
                result.name,
                result.processing_status,
                result.processing_message,
            )

            if fail_on_point_error:
                raise RuntimeError(
                    f"Обработка точки '{result.name}' остановлена: "
                    f"{result.processing_message}"
                )

        self._analyze_deformations(
            points_epoch1=valid_points_epoch1,
            points_epoch2=valid_points_epoch2,
        )

        return self.to_dataframe()

    def _process_reference_point(
        self,
        reference_point: ReferencePoint,
        neighborhood_epoch1: Scan,
        neighborhood_epoch2: Scan,
    ) -> ReferencePointProcessingResult:
        radius = (
            float(reference_point.radius)
            if reference_point.radius is not None
            else self.default_radius
        )

        result = ReferencePointProcessingResult(
            name=reference_point.name,
            reference_x=float(reference_point.x),
            reference_y=float(reference_point.y),
            reference_z=float(reference_point.z),
            radius=radius,
            epoch1_points=len(neighborhood_epoch1),
            epoch2_points=len(neighborhood_epoch2),
        )

        try:
            self._validate_neighborhoods(result)

            point_epoch1 = self._extract_virtual_point(
                neighborhood=neighborhood_epoch1,
                reference_point=reference_point,
                epoch_name="epoch1",
            )

            point_epoch2 = self._extract_virtual_point(
                neighborhood=neighborhood_epoch2,
                reference_point=reference_point,
                epoch_name="epoch2",
            )

            # Сохраняем координаты для диагностики даже при UNRELIABLE,
            # но передаём в DeformationAnalyzer только SUCCESS.
            result.point_epoch1 = point_epoch1
            result.point_epoch2 = point_epoch2

            self._validate_virtual_point_quality(
                virtual_point=point_epoch1,
                epoch_name="epoch1",
            )
            self._validate_virtual_point_quality(
                virtual_point=point_epoch2,
                epoch_name="epoch2",
            )

            result.epoch1_reference_distance = (
                self._distance_to_reference(
                    virtual_point=point_epoch1,
                    reference_point=reference_point,
                )
            )
            result.epoch2_reference_distance = (
                self._distance_to_reference(
                    virtual_point=point_epoch2,
                    reference_point=reference_point,
                )
            )

            max_reference_distance = (
                radius
                * self.max_reference_distance_factor
            )

            if (
                result.epoch1_reference_distance
                > max_reference_distance
            ):
                raise ValueError(
                    "Виртуальная точка epoch1 удалена от опорной точки "
                    f"на {result.epoch1_reference_distance:.6f} м; "
                    f"допуск {max_reference_distance:.6f} м."
                )

            if (
                result.epoch2_reference_distance
                > max_reference_distance
            ):
                raise ValueError(
                    "Виртуальная точка epoch2 удалена от опорной точки "
                    f"на {result.epoch2_reference_distance:.6f} м; "
                    f"допуск {max_reference_distance:.6f} м."
                )

            result.pair_distance = self._distance_between_points(
                point_epoch1,
                point_epoch2,
            )

            if result.pair_distance > self.max_pair_distance:
                raise ValueError(
                    "Виртуальные точки epoch1 и epoch2 расходятся на "
                    f"{result.pair_distance:.6f} м; "
                    f"контрольный предел {self.max_pair_distance:.6f} м. "
                    "Вероятна нестабильная сегментация или выбор разных "
                    "конструктивных поверхностей."
                )

            result.processing_status = self.SUCCESS
            result.processing_message = "OK"

        except ValueError as error:
            result.processing_status = self.UNRELIABLE
            result.processing_message = str(error)

        except Exception as error:
            result.processing_status = self.FAILED
            result.processing_message = (
                f"{type(error).__name__}: {error}"
            )

            logger.exception(
                "Техническая ошибка обработки точки '{}'",
                reference_point.name,
            )

        return result

    def _validate_neighborhoods(
        self,
        result: ReferencePointProcessingResult,
    ) -> None:
        if (
            result.epoch1_points is None
            or result.epoch2_points is None
        ):
            raise RuntimeError(
                "Не определено число точек в окрестностях."
            )

        if result.epoch1_points < self.min_neighborhood_points:
            raise ValueError(
                f"Недостаточно точек в epoch1: "
                f"{result.epoch1_points}; требуется не менее "
                f"{self.min_neighborhood_points}."
            )

        if result.epoch2_points < self.min_neighborhood_points:
            raise ValueError(
                f"Недостаточно точек в epoch2: "
                f"{result.epoch2_points}; требуется не менее "
                f"{self.min_neighborhood_points}."
            )

    @staticmethod
    def _validate_virtual_point_quality(
        virtual_point: Any,
        epoch_name: str,
    ) -> None:
        """
        Не допускает к сравнению точки, рассчитанные при неустойчивой
        геометрии или без ковариационной оценки.
        """
        if not virtual_point.reliable_accuracy:
            raise ValueError(
                f"Виртуальная точка {epoch_name} ненадёжна: "
                f"geometry_status={virtual_point.status}."
            )

        if virtual_point.cov_xyz is None:
            raise ValueError(
                f"Для виртуальной точки {epoch_name} отсутствует "
                "ковариационная матрица."
            )

        covariance = np.asarray(
            virtual_point.cov_xyz,
            dtype=np.float64,
        )

        if covariance.shape != (3, 3):
            raise ValueError(
                f"Ковариация точки {epoch_name} имеет неверную форму "
                f"{covariance.shape}; ожидается (3, 3)."
            )

        if not np.all(np.isfinite(covariance)):
            raise ValueError(
                f"Ковариация точки {epoch_name} содержит NaN или Inf."
            )

        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)

        if np.any(eigenvalues < -1e-12):
            raise ValueError(
                f"Ковариация точки {epoch_name} не является положительно "
                "полуопределённой."
            )

        if not np.all(
            np.isfinite([
                virtual_point.x,
                virtual_point.y,
                virtual_point.z,
            ])
        ):
            raise ValueError(
                f"Координаты виртуальной точки {epoch_name} "
                "содержат NaN или Inf."
            )

    @staticmethod
    def _distance_to_reference(
        virtual_point: Any,
        reference_point: ReferencePoint,
    ) -> float:
        virtual_xyz = np.asarray(
            [
                virtual_point.x,
                virtual_point.y,
                virtual_point.z,
            ],
            dtype=np.float64,
        )

        reference_xyz = np.asarray(
            reference_point.as_array(),
            dtype=np.float64,
        )

        return float(
            np.linalg.norm(virtual_xyz - reference_xyz)
        )

    @staticmethod
    def _distance_between_points(
        point_epoch1: Any,
        point_epoch2: Any,
    ) -> float:
        xyz_epoch1 = np.asarray(
            [
                point_epoch1.x,
                point_epoch1.y,
                point_epoch1.z,
            ],
            dtype=np.float64,
        )

        xyz_epoch2 = np.asarray(
            [
                point_epoch2.x,
                point_epoch2.y,
                point_epoch2.z,
            ],
            dtype=np.float64,
        )

        return float(
            np.linalg.norm(xyz_epoch2 - xyz_epoch1)
        )

    def _extract_virtual_point(
        self,
        neighborhood: Scan,
        reference_point: ReferencePoint,
        epoch_name: str,
    ) -> Any:
        logger.info(
            "Выделение виртуальной точки '{}', {}: точек={}",
            reference_point.name,
            epoch_name,
            len(neighborhood),
        )

        exacter = CrossPointExacter.from_scan(
            scan=neighborhood,
            reference_xyz=reference_point.as_array(),
            show_scans=False,
            normal_k=self.normal_k,
            min_points_per_plane=self.min_points_per_plane,
            cluster_eps=self.cluster_eps,
            cluster_min_samples=self.cluster_min_samples,
        )

        exacter.calculate_planes()
        virtual_point = exacter.calculate_intersect_point()
        virtual_point.name = reference_point.name

        logger.info(
            "Виртуальная точка '{}', {}: "
            "X={:.6f}, Y={:.6f}, Z={:.6f}, geometry_status={}",
            reference_point.name,
            epoch_name,
            virtual_point.x,
            virtual_point.y,
            virtual_point.z,
            virtual_point.status,
        )

        return virtual_point

    def _analyze_deformations(
        self,
        points_epoch1: list[Any],
        points_epoch2: list[Any],
    ) -> None:
        """
        Передаёт в DeformationAnalyzer исключительно пары,
        прошедшие геометрический контроль.
        """
        if not points_epoch1:
            self.deformation_results = []

            logger.warning(
                "Нет валидных пар виртуальных точек для анализа деформаций."
            )
            return

        logger.info(
            "Статистический анализ валидных пар: {}",
            len(points_epoch1),
        )

        analyzer = DeformationAnalyzer(alpha=self.alpha)
        analyzer.analyze_point_sets(
            points_epoch1=points_epoch1,
            points_epoch2=points_epoch2,
        )

        self.deformation_results = analyzer.results

        logger.success(
            "Статистический анализ завершён: пар={}, значимых={}",
            len(analyzer.results),
            analyzer.n_significant,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Формирует CSV-таблицу с опорными точками, диагностикой извлечения,
        координатами виртуальных точек и — только для SUCCESS — деформациями.
        """
        rows: dict[str, dict[str, Any]] = {
            result.name: result.as_dict()
            for result in self.processing_results
        }

        for deformation in self.deformation_results:
            row = rows.get(deformation.name)

            if row is None:
                continue

            if row["processing_status"] != self.SUCCESS:
                continue

            row.update({
                "dx": float(deformation.delta[0]),
                "dy": float(deformation.delta[1]),
                "dz": float(deformation.delta[2]),
                "displacement": float(deformation.displacement),
                "displacement_mm": float(
                    deformation.displacement_mm
                ),
                "sigma_dx": float(deformation.sigma_dx),
                "sigma_dy": float(deformation.sigma_dy),
                "sigma_dz": float(deformation.sigma_dz),
                "sigma_displacement": float(
                    deformation.sigma_displacement
                ),
                "sigma_displacement_mm": float(
                    deformation.sigma_displacement_mm
                ),
                "t_value": float(deformation.t_value),
                "p_value_t": float(deformation.p_value_t),
                "significant_t": bool(deformation.significant_t),
                "chi2_value": (
                    float(deformation.chi2_value)
                    if deformation.chi2_value is not None
                    else np.nan
                ),
                "p_value_chi2": (
                    float(deformation.p_value_chi2)
                    if deformation.p_value_chi2 is not None
                    else np.nan
                ),
                "significant_chi2": deformation.significant_chi2,
                "analysis_reliable": bool(deformation.reliable),
            })

        dataframe = pd.DataFrame(list(rows.values()))

        if dataframe.empty:
            return dataframe

        return dataframe.sort_values(
            by="name",
            kind="stable",
        ).reset_index(drop=True)

    def to_csv(
        self,
        output_path: str | Path,
        index: bool = False,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe = self.to_dataframe()

        dataframe.to_csv(
            output_path,
            index=index,
            encoding="utf-8",
        )

        logger.success(
            "Итоговая таблица сохранена: {}",
            output_path,
        )
