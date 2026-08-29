from __future__ import annotations

from pathlib import Path

import laspy
from tqdm import tqdm

from CONFIG import POINTS_CHUNK_COUNT
from app.scan.ScanPoint import ScanPoint
from app.scan.parsers.ScanParserABC import ScanParserABC


class ScanParserFromLas(ScanParserABC):
    """
    Загружает LAS-файл в Scan порциями с отображением прогресса.

    Прогресс показывается по реальному количеству точек, указанному
    в LAS-header.point_count.
    """

    def __init__(
        self,
        file_path: str | Path,
        chunk_count: int = POINTS_CHUNK_COUNT,
        show_progress: bool = True,
    ) -> None:
        super().__init__(str(file_path))

        if chunk_count < 1:
            raise ValueError(
                "chunk_count должен быть положительным."
            )

        self.chunk_count = int(chunk_count)
        self.show_progress = bool(show_progress)

    @staticmethod
    def _get_xyz(
        raw_xyz: tuple[int, int, int],
        scales,
        offsets,
    ) -> tuple[float, float, float]:
        """
        Преобразует целочисленные LAS-координаты в метрические:

            coordinate = raw * scale + offset
        """
        return tuple(
            float(raw * scale + offset)
            for raw, scale, offset in zip(
                raw_xyz,
                scales,
                offsets,
            )
        )

    @staticmethod
    def _get_rgb(
        raw_rgb: tuple[int, int, int] | None,
    ) -> tuple[int, int, int]:
        """
        Преобразует 16-битный LAS RGB в 8-битный RGB.
        """
        if raw_rgb is None:
            return 0, 0, 0

        return tuple(
            int(value // 256)
            for value in raw_rgb
        )

    def parse(self, scan) -> None:
        """
        Читает LAS-файл блоками и добавляет точки в Scan.
        """
        input_path = Path(self.file_path)

        with laspy.open(input_path) as input_las:
            total_points = int(
                input_las.header.point_count
            )

            point_format = input_las.header.point_format
            dimension_names = set(
                point_format.dimension_names
            )

            has_rgb = {
                "red",
                "green",
                "blue",
            }.issubset(dimension_names)

            progress = tqdm(
                total=total_points,
                desc=f"Загрузка {input_path.name}",
                unit="точка",
                dynamic_ncols=True,
                disable=not self.show_progress,
            )

            try:
                for chunk in input_las.chunk_iterator(
                    self.chunk_count
                ):
                    raw_points = chunk.array
                    scales = chunk.scales
                    offsets = chunk.offsets

                    for raw_point in raw_points:
                        x_coord, y_coord, z_coord = (
                            self._get_xyz(
                                raw_xyz=(
                                    raw_point["X"],
                                    raw_point["Y"],
                                    raw_point["Z"],
                                ),
                                scales=scales,
                                offsets=offsets,
                            )
                        )

                        rgb = (
                            self._get_rgb(
                                raw_rgb=(
                                    raw_point["red"],
                                    raw_point["green"],
                                    raw_point["blue"],
                                )
                            )
                            if has_rgb
                            else (0, 0, 0)
                        )

                        scan.add_point(
                            ScanPoint(
                                x=x_coord,
                                y=y_coord,
                                z=z_coord,
                                color=rgb,
                            )
                        )

                    progress.update(
                        len(raw_points)
                    )

            finally:
                progress.close()