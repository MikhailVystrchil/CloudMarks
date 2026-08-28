from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.scan.Scan import Scan


class ScanSplitterByLabels:
    """
    Разделяет Scan на подсканы по метке точки.

    По умолчанию метка хранится в ``ScanPoint.labels``.
    Метка DBSCAN ``-1`` считается шумом и исключается, если
    ``include_noise=False``.
    """

    def __init__(
        self,
        scan: Scan,
        *,
        label_attr: str = "labels",
        include_noise: bool = False,
        noise_label: int = -1,
        copy_points: bool = False,
    ) -> None:
        self.scan = scan
        self.label_attr = label_attr
        self.include_noise = include_noise
        self.noise_label = noise_label
        self.copy_points = copy_points

    def split(self) -> dict[int, Scan]:
        """
        Возвращает словарь ``{label: Scan}``.

        Точки без атрибута `label_attr` игнорируются.
        """
        # Локальный импорт выполняется только после полной инициализации
        # app.scan.Scan, поэтому циклического импорта нет.
        from app.scan.Scan import Scan

        points_by_label: dict[int, list] = defaultdict(list)

        for point in self.scan:
            if not hasattr(point, self.label_attr):
                continue

            label = int(
                getattr(point, self.label_attr)
            )

            if (
                label == self.noise_label
                and not self.include_noise
            ):
                continue

            points_by_label[label].append(point)

        return {
            label: Scan.from_points(
                scan_name=f"{self.scan.name}_label_{label}",
                points=points,
                copy_points=self.copy_points,
                include_normals=True,
                include_labels=True,
            )
            for label, points in sorted(
                points_by_label.items()
            )
        }
