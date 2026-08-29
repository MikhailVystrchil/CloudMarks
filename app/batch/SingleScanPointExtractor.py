from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from app.batch.ReferencePoint import ReferencePoint
from app.batch.ReferencePointReader import ReferencePointReader
from app.batch.ScanNeighborhoodExtractor import ScanNeighborhoodExtractor
from app.cross_points.CrossPointExacter import CrossPointExacter
from app.scan.Scan import Scan


@dataclass(slots=True)
class SingleScanPointResult:
    """
    Результат извлечения одной виртуальной точки из скана.

    status:
        SUCCESS    — точка вычислена и прошла контроль качества;
        UNRELIABLE — координаты получены, но геометрия или точность
                     не прошли проверку;
        FAILED     — извлечь точку не удалось из-за технической ошибки.
    """

    name: str
    reference_x: float
    reference_y: float
    reference_z: float
    radius: float

    neighborhood_points: int | None = None
    reference_distance: float | None = None

    status: str = "FAILED"
    message: str = ""

    point: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        """
        Преобразует результат в плоскую структуру для DataFrame и CSV.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "reference_x": self.reference_x,
            "reference_y": self.reference_y,
            "reference_z": self.reference_z,
            "radius": self.radius,
            "neighborhood_points": self.neighborhood_points,
            "reference_distance": self.reference_distance,
            "status": self.status,
            "message": self.message,
        }

        point = self.point

        if point is None:
            result.update(
                {
                    "x": np.nan,
                    "y": np.nan,
                    "z": np.nan,
                    "geometry_status": None,
                    "reliable_accuracy": False,
                    "sigma_x": np.nan,
                    "sigma_y": np.nan,
                    "sigma_z": np.nan,
                }
            )
            return result

        sigma = (
            point.sigma_xyz
            if point.reliable_accuracy and point.sigma_xyz is not None
            else (np.nan, np.nan, np.nan)
        )

        result.update(
            {
                "x": float(point.x),
                "y": float(point.y),
                "z": float(point.z),
                "geometry_status": point.status,
                "reliable_accuracy": bool(point.reliable_accuracy),
                "sigma_x": float(sigma[0]),
                "sigma_y": float(sigma[1]),
                "sigma_z": float(sigma[2]),
            }
        )

        return result


class SingleScanPointExtractor:
    """
    Извлекает набор виртуальных точек из одного скана по опорным координатам.

    Последовательность обработки:

    1. Загружается единый скан LAS/TXT.
    2. Для него один раз строится cKDTree.
    3. Для каждой опорной точки вырезается сферическая окрестность.
    4. В окрестности сегментируются три локальные плоскости.
    5. Рассчитывается виртуальная точка — пересечение плоскостей.
    6. Проводится контроль качества геометрии и расстояния до опорной точки.
    7. Результаты доступны как DataFrame, консольный отчёт и CSV.

    Пример
    -------
    >>> extractor = SingleScanPointExtractor.from_files(
    ...     scan_path="data/scan.las",
    ...     reference_points_path="data/marks.txt",
    ...     default_radius=0.15,
    ... )
    >>> extractor.run()
    >>> extractor.print_report()
    >>> extractor.export_csv("output/points.csv")
    """

    SUCCESS = "SUCCESS"
    UNRELIABLE = "UNRELIABLE"
    FAILED = "FAILED"

    def __init__(
        self,
        scan: Scan,
        default_radius: float,
        min_neighborhood_points: int = 60,
        min_points_per_plane: int = 15,
        max_reference_distance_factor: float = 1.25,
        normal_k: int = 12,
        cluster_eps: float = 0.08,
        cluster_min_samples: int = 3,
    ) -> None:
        if default_radius <= 0:
            raise ValueError(
                "default_radius должен быть положительным."
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

        self.scan = scan

        self.default_radius = float(default_radius)
        self.min_neighborhood_points = int(
            min_neighborhood_points
        )
        self.min_points_per_plane = int(
            min_points_per_plane
        )
        self.max_reference_distance_factor = float(
            max_reference_distance_factor
        )
        self.normal_k = int(normal_k)
        self.cluster_eps = float(cluster_eps)
        self.cluster_min_samples = int(
            cluster_min_samples
        )

        self.extractor = ScanNeighborhoodExtractor(
            scan=scan
        )
        self.results: list[SingleScanPointResult] = []

    @classmethod
    def from_files(
            cls,
            scan_path: str | Path,
            reference_points_path: str | Path,
            default_radius: float,
            show_progress: bool = True,
            **kwargs: Any,
    ) -> "SingleScanPointExtractor":
        """
        Создаёт обработчик, загружая скан и файл опорных точек.
        """
        if show_progress:
            print("[1/3] Загрузка скана...", end=" ", flush=True)

        scan = cls._load_scan(
            file_path=scan_path
        )

        if show_progress:
            print(f"готово: {len(scan):,} точек")

        if show_progress:
            print("[2/3] Построение пространственного индекса...", end=" ", flush=True)

        instance = cls(
            scan=scan,
            default_radius=default_radius,
            **kwargs,
        )

        if show_progress:
            print("готово")

        instance.reference_points = ReferencePointReader.read(
            reference_points_path
        )

        if show_progress:
            print(
                f"Опорных точек: "
                f"{len(instance.reference_points)}"
            )

        return instance

    @staticmethod
    def _load_scan(
        file_path: str | Path,
    ) -> Scan:
        """
        Загружает один скан и проверяет, что в нём есть точки.
        """
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Файл скана не найден: {path}"
            )

        logger.info(
            "Загрузка скана: {}",
            path,
        )

        scan = Scan(path.stem)
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
            scan.name,
            len(scan),
        )

        return scan

    def run(
            self,
            reference_points: list[ReferencePoint] | None = None,
            fail_on_point_error: bool = False,
            show_progress: bool = True,
    ) -> "SingleScanPointExtractor":
        """
        Обрабатывает все опорные точки.
        """
        points = reference_points or getattr(
            self,
            "reference_points",
            None,
        )

        if not points:
            raise ValueError(
                "Список опорных точек пуст."
            )

        names = [point.name for point in points]

        if len(names) != len(set(names)):
            raise ValueError(
                "Имена опорных точек должны быть уникальными."
            )

        self.results = []

        iterator = points

        if show_progress:
            iterator = tqdm(
                points,
                desc="[3/3] Извлечение точек",
                unit="точка",
                dynamic_ncols=True,
            )

        for point in iterator:
            result = self._process_reference_point(
                reference_point=point,
                fail_on_point_error=fail_on_point_error,
            )

            self.results.append(result)

        success_count = sum(
            result.status == self.SUCCESS
            for result in self.results
        )

        if show_progress:
            print(
                f"Готово: {success_count}/{len(self.results)} "
                "надёжных точек"
            )

        return self

    def _process_reference_point(
        self,
        reference_point: ReferencePoint,
        fail_on_point_error: bool,
    ) -> SingleScanPointResult:
        """
        Обрабатывает одну опорную точку и возвращает диагностический результат.
        """
        radius = (
            float(reference_point.radius)
            if reference_point.radius is not None
            else self.default_radius
        )

        result = SingleScanPointResult(
            name=reference_point.name,
            reference_x=float(reference_point.x),
            reference_y=float(reference_point.y),
            reference_z=float(reference_point.z),
            radius=radius,
        )

        try:
            neighborhood = self.extractor.extract_sphere(
                reference_point=reference_point,
                radius=radius,
            )

            result.neighborhood_points = len(neighborhood)

            if result.neighborhood_points < self.min_neighborhood_points:
                raise ValueError(
                    "Недостаточно точек в окрестности: "
                    f"{result.neighborhood_points}; требуется не менее "
                    f"{self.min_neighborhood_points}."
                )

            virtual_point = self._extract_virtual_point(
                neighborhood=neighborhood,
                reference_point=reference_point,
            )
            result.point = virtual_point

            if not virtual_point.reliable_accuracy:
                raise ValueError(
                    "Точка ненадёжна: "
                    f"geometry_status={virtual_point.status}."
                )

            result.reference_distance = float(
                np.linalg.norm(
                    virtual_point.as_array()
                    - reference_point.as_array()
                )
            )

            max_distance = (
                radius
                * self.max_reference_distance_factor
            )

            if result.reference_distance > max_distance:
                raise ValueError(
                    "Виртуальная точка удалена от опорной точки на "
                    f"{result.reference_distance:.6f} м; "
                    f"допуск {max_distance:.6f} м."
                )

            result.status = self.SUCCESS
            result.message = "OK"

        except ValueError as error:
            result.status = self.UNRELIABLE
            result.message = str(error)

        except Exception as error:
            result.status = self.FAILED
            result.message = (
                f"{type(error).__name__}: {error}"
            )

            logger.exception(
                "Техническая ошибка обработки точки '{}'",
                reference_point.name,
            )

            if fail_on_point_error:
                raise

        if result.status != self.SUCCESS:
            logger.warning(
                "Точка '{}' исключена: status={}, причина={}",
                result.name,
                result.status,
                result.message,
            )

        return result

    def _extract_virtual_point(
        self,
        neighborhood: Scan,
        reference_point: ReferencePoint,
    ) -> Any:
        """
        Вычисляет виртуальную точку из уже выделенной локальной окрестности.
        """
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
            "Виртуальная точка '{}': "
            "X={:.6f}, Y={:.6f}, Z={:.6f}, geometry_status={}",
            reference_point.name,
            virtual_point.x,
            virtual_point.y,
            virtual_point.z,
            virtual_point.status,
        )

        return virtual_point

    def to_dataframe(self) -> pd.DataFrame:
        """
        Возвращает результаты как DataFrame, отсортированный по имени точки.
        """
        dataframe = pd.DataFrame(
            [
                result.as_dict()
                for result in self.results
            ]
        )

        if dataframe.empty:
            return dataframe

        return dataframe.sort_values(
            by="name",
            kind="stable",
        ).reset_index(drop=True)

    def print_report(self) -> None:
        """
        Печатает компактную сводку результатов в стандартный вывод.

        В таблице выводятся:
        - имя точки;
        - рассчитанные X, Y, Z;
        - СКП координат;
        - статус обработки.
        """
        if not self.results:
            print(
                "Нет результатов: сначала вызовите run()."
            )
            return

        dataframe = self.to_dataframe()

        success_count = int(
            (
                dataframe["status"] == self.SUCCESS
            ).sum()
        )

        print(f"Скан: {self.scan.name}")
        print(
            f"Опорных точек: {len(dataframe)}, "
            f"успешно: {success_count}"
        )
        print("-" * 96)

        columns = [
            "name",
            "x",
            "y",
            "z",
            "sigma_x",
            "sigma_y",
            "sigma_z",
            "status",
        ]

        print(
            dataframe[columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

        print("-" * 96)

    def export_csv(
        self,
        output_path: str | Path,
        index: bool = False,
    ) -> Path:
        """
        Экспортирует полную таблицу результатов в UTF-8 CSV.

        Возвращает путь к созданному файлу.
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

        logger.success(
            "Результаты сохранены: {}",
            output_path,
        )

        return output_path
