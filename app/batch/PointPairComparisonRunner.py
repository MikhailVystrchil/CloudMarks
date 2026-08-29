from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from app.batch.ExtractionConfig import ExtractionConfig
from app.batch.ReferencePoint import ReferencePoint
from app.batch.ReferencePointReader import ReferencePointReader
from app.batch.SingleScanPointExtractor import (
    SingleScanPointExtractor,
    SingleScanPointResult,
)
from app.batch.scan_loading import load_scan_from_file
from app.batch.validation import ensure_unique_names
from app.deformation.DeformationAnalyzer import DeformationAnalyzer
from app.scan.Scan import Scan


@dataclass(slots=True)
class ReferencePointProcessingResult:
    """
    Результат обработки одной опорной точки для двух эпох.

    processing_status:
        SUCCESS    — точки успешно извлечены в обеих эпохах и прошли
                     межэпоховый контроль;
        UNRELIABLE — точка была вычислена, но не прошла один из
                     контролей качества;
        FAILED     — произошла техническая ошибка извлечения.
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
        """
        Возвращает плоское представление результата для DataFrame/CSV.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "reference_x": self.reference_x,
            "reference_y": self.reference_y,
            "reference_z": self.reference_z,
            "radius": self.radius,
            "epoch1_neighborhood_points": self.epoch1_points,
            "epoch2_neighborhood_points": self.epoch2_points,
            "epoch1_reference_distance": (
                self.epoch1_reference_distance
            ),
            "epoch2_reference_distance": (
                self.epoch2_reference_distance
            ),
            "pair_distance": self.pair_distance,
            "processing_status": self.processing_status,
            "processing_message": self.processing_message,
        }

        self._append_point(
            result=result,
            point=self.point_epoch1,
            prefix="epoch1",
        )

        self._append_point(
            result=result,
            point=self.point_epoch2,
            prefix="epoch2",
        )

        return result

    @staticmethod
    def _append_point(
        result: dict[str, Any],
        point: Any | None,
        prefix: str,
    ) -> None:
        """
        Добавляет параметры виртуальной точки одной эпохи в словарь.
        """
        if point is None:
            result.update(
                {
                    f"{prefix}_x": np.nan,
                    f"{prefix}_y": np.nan,
                    f"{prefix}_z": np.nan,
                    f"{prefix}_status": None,
                    f"{prefix}_reliable_accuracy": False,
                    f"{prefix}_sigma_x": np.nan,
                    f"{prefix}_sigma_y": np.nan,
                    f"{prefix}_sigma_z": np.nan,
                }
            )
            return

        result.update(
            point.as_flat_fields(prefix=prefix)
        )


class PointPairComparisonRunner:
    """
    Сравнивает два зарегистрированных облака точек.

    Локальная обработка каждой эпохи делегирована
    ``SingleScanPointExtractor``. Этот класс выполняет только:

    1. запуск извлечения точек для epoch1 и epoch2;
    2. объединение результатов по имени опорной точки;
    3. контроль расстояния между одноимёнными виртуальными точками;
    4. статистический анализ деформаций для принятых пар.

    Сканы должны быть предварительно зарегистрированы в общей
    системе координат.
    """

    SUCCESS = "SUCCESS"
    UNRELIABLE = "UNRELIABLE"
    FAILED = "FAILED"

    def __init__(
        self,
        scan_epoch1: Scan,
        scan_epoch2: Scan,
        config: ExtractionConfig,
        *,
        alpha: float = 0.05,
        max_pair_distance: float | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                "alpha должен принадлежать интервалу (0, 1)."
            )

        self.scan_epoch1 = scan_epoch1
        self.scan_epoch2 = scan_epoch2
        self.config = config
        self.alpha = float(alpha)

        self.max_pair_distance = (
            float(max_pair_distance)
            if max_pair_distance is not None
            else self.config.default_radius * 0.05
        )

        if self.max_pair_distance <= 0.0:
            raise ValueError(
                "max_pair_distance должен быть положительным."
            )

        self.extractor_epoch1 = self._create_single_extractor(
            scan=scan_epoch1
        )
        self.extractor_epoch2 = self._create_single_extractor(
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
        config: ExtractionConfig,
        *,
        alpha: float = 0.05,
        max_pair_distance: float | None = None,
    ) -> "PointPairComparisonRunner":
        """
        Загружает два скана и создаёт runner для их сравнения.
        """
        scan_epoch1 = load_scan_from_file(
            epoch1_path,
            scan_name="epoch1",
        )
        scan_epoch2 = load_scan_from_file(
            epoch2_path,
            scan_name="epoch2",
        )

        return cls(
            scan_epoch1=scan_epoch1,
            scan_epoch2=scan_epoch2,
            config=config,
            alpha=alpha,
            max_pair_distance=max_pair_distance,
        )

    def _create_single_extractor(
        self,
        scan: Scan,
    ) -> SingleScanPointExtractor:
        """
        Создаёт извлекатель точек одной эпохи с общей конфигурацией.
        """
        return SingleScanPointExtractor(
            scan=scan,
            config=self.config,
        )

    def run_from_reference_file(
        self,
        reference_points_path: str | Path,
        *,
        workers: int = -1,
        fail_on_point_error: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Читает опорные точки из файла и запускает сравнение.
        """
        reference_points = ReferencePointReader.read(
            reference_points_path
        )

        return self.run(
            reference_points=reference_points,
            workers=workers,
            fail_on_point_error=fail_on_point_error,
            show_progress=show_progress,
        )

    def run(
        self,
        reference_points: list[ReferencePoint],
        *,
        workers: int = -1,
        fail_on_point_error: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Обрабатывает опорные точки в двух эпохах.

        Используется единый progress bar с total = 2 * N:
        один шаг — обработка одной опорной точки в одной эпохе.
        """
        if not reference_points:
            raise ValueError(
                "Список опорных точек пуст."
            )

        if workers == 0 or workers < -1:
            raise ValueError(
                "workers должен быть -1 или положительным целым числом."
            )

        ensure_unique_names(
            (reference_point.name for reference_point in reference_points),
            entity_name="опорных точек",
        )

        self.processing_results = []
        self.deformation_results = []

        progress = tqdm(
            total=2 * len(reference_points),
            desc="Обработка точек",
            unit="эпоха",
            dynamic_ncols=True,
            disable=not show_progress,
        )

        valid_points_epoch1: list[Any] = []
        valid_points_epoch2: list[Any] = []

        try:
            for reference_point in reference_points:
                result_epoch1 = (
                    self.extractor_epoch1._process_reference_point(
                        reference_point=reference_point,
                        fail_on_point_error=fail_on_point_error,
                    )
                )
                progress.update(1)

                result_epoch2 = (
                    self.extractor_epoch2._process_reference_point(
                        reference_point=reference_point,
                        fail_on_point_error=fail_on_point_error,
                    )
                )
                progress.update(1)

                comparison_result = self._combine_epoch_results(
                    reference_point=reference_point,
                    result_epoch1=result_epoch1,
                    result_epoch2=result_epoch2,
                )

                self.processing_results.append(comparison_result)

                if comparison_result.processing_status == self.SUCCESS:
                    valid_points_epoch1.append(
                        comparison_result.point_epoch1
                    )
                    valid_points_epoch2.append(
                        comparison_result.point_epoch2
                    )

                elif fail_on_point_error:
                    raise RuntimeError(
                        f"Точка '{comparison_result.name}' отклонена: "
                        f"{comparison_result.processing_message}"
                    )

                progress.set_postfix_str(
                    f"OK={len(valid_points_epoch1)}"
                )

        finally:
            progress.close()

        if show_progress:
            tqdm.write("Анализ деформаций...")

        self._analyze_deformations(
            points_epoch1=valid_points_epoch1,
            points_epoch2=valid_points_epoch2,
        )

        if show_progress:
            tqdm.write(
                f"Готово: {len(valid_points_epoch1)}/"
                f"{len(reference_points)} пар"
            )

        return self.to_dataframe()

    def _combine_epoch_results(
        self,
        reference_point: ReferencePoint,
        result_epoch1: SingleScanPointResult,
        result_epoch2: SingleScanPointResult,
    ) -> ReferencePointProcessingResult:
        """
        Объединяет результаты одной точки в двух эпохах.
        """
        radius = (
            float(reference_point.radius)
            if reference_point.radius is not None
            else self.config.default_radius
        )

        result = ReferencePointProcessingResult(
            name=reference_point.name,
            reference_x=float(reference_point.x),
            reference_y=float(reference_point.y),
            reference_z=float(reference_point.z),
            radius=radius,
            epoch1_points=result_epoch1.neighborhood_points,
            epoch2_points=result_epoch2.neighborhood_points,
            epoch1_reference_distance=(
                result_epoch1.reference_distance
            ),
            epoch2_reference_distance=(
                result_epoch2.reference_distance
            ),
            point_epoch1=result_epoch1.point,
            point_epoch2=result_epoch2.point,
        )

        if result_epoch1.status != SingleScanPointExtractor.SUCCESS:
            result.processing_status = self._map_single_status(
                result_epoch1.status
            )
            result.processing_message = (
                f"epoch1: {result_epoch1.message}"
            )
            return result

        if result_epoch2.status != SingleScanPointExtractor.SUCCESS:
            result.processing_status = self._map_single_status(
                result_epoch2.status
            )
            result.processing_message = (
                f"epoch2: {result_epoch2.message}"
            )
            return result

        if (
            result_epoch1.point is None
            or result_epoch2.point is None
        ):
            result.processing_status = self.FAILED
            result.processing_message = (
                "Одиночный извлекатель вернул SUCCESS, "
                "но виртуальная точка отсутствует."
            )
            return result

        result.pair_distance = self._distance_between_points(
            point_epoch1=result_epoch1.point,
            point_epoch2=result_epoch2.point,
        )

        if result.pair_distance > self.max_pair_distance:
            result.processing_status = self.UNRELIABLE
            result.processing_message = (
                "Виртуальные точки epoch1 и epoch2 расходятся на "
                f"{result.pair_distance:.6f} м; "
                f"контрольный предел "
                f"{self.max_pair_distance:.6f} м."
            )
            return result

        result.processing_status = self.SUCCESS
        result.processing_message = "OK"

        return result

    @staticmethod
    def _map_single_status(
        status: str,
    ) -> str:
        """
        Приводит статус одиночного извлечения к статусу пары эпох.
        """
        if status == SingleScanPointExtractor.UNRELIABLE:
            return PointPairComparisonRunner.UNRELIABLE

        return PointPairComparisonRunner.FAILED

    @staticmethod
    def _distance_between_points(
        point_epoch1: Any,
        point_epoch2: Any,
    ) -> float:
        """
        Возвращает расстояние между виртуальными точками двух эпох.
        """
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

    def _analyze_deformations(
        self,
        points_epoch1: list[Any],
        points_epoch2: list[Any],
    ) -> None:
        """
        Анализирует только пары, прошедшие весь геометрический контроль.
        """
        if not points_epoch1:
            self.deformation_results = []
            return

        analyzer = DeformationAnalyzer(alpha=self.alpha)

        analyzer.analyze_point_sets(
            points_epoch1=points_epoch1,
            points_epoch2=points_epoch2,
        )

        self.deformation_results = analyzer.results

    def to_dataframe(self) -> pd.DataFrame:
        """
        Возвращает полную таблицу с диагностикой и деформациями.
        """
        rows = {
            result.name: result.as_dict()
            for result in self.processing_results
        }

        for deformation in self.deformation_results:
            row = rows.get(deformation.name)

            if row is None:
                continue

            if row["processing_status"] != self.SUCCESS:
                continue

            row.update(
                {
                    "dx": float(deformation.delta[0]),
                    "dy": float(deformation.delta[1]),
                    "dz": float(deformation.delta[2]),
                    "displacement": float(
                        deformation.displacement
                    ),
                    "displacement_mm": float(
                        deformation.displacement_mm
                    ),
                    "sigma_dx": float(
                        deformation.sigma_dx
                    ),
                    "sigma_dy": float(
                        deformation.sigma_dy
                    ),
                    "sigma_dz": float(
                        deformation.sigma_dz
                    ),
                    "sigma_displacement": float(
                        deformation.sigma_displacement
                    ),
                    "sigma_displacement_mm": float(
                        deformation.sigma_displacement_mm
                    ),
                    "t_value": float(
                        deformation.t_value
                    ),
                    "p_value_t": float(
                        deformation.p_value_t
                    ),
                    "significant_t": bool(
                        deformation.significant_t
                    ),
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
                    "significant_chi2": (
                        deformation.significant_chi2
                    ),
                    "analysis_reliable": bool(
                        deformation.reliable
                    ),
                }
            )

        dataframe = pd.DataFrame(
            list(rows.values())
        )

        if dataframe.empty:
            return dataframe

        return dataframe.sort_values(
            by="name",
            kind="stable",
        ).reset_index(drop=True)

    def to_csv(
        self,
        output_path: str | Path,
        *,
        index: bool = False,
    ) -> None:
        """
        Экспортирует полную диагностическую таблицу в CSV.
        """
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.to_dataframe().to_csv(
            output_path,
            index=index,
            encoding="utf-8",
        )
