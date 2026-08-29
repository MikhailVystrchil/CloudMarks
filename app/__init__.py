"""
CloudMarks.

Инструменты безмарочного деформационного мониторинга сооружений
по данным наземного лазерного сканирования.

Основной вычислительный конвейер:

1. Извлечение локальной окрестности около опорной точки.
2. Вычисление нормалей и сегментация точек по направлениям нормалей.
3. Выделение трёх конструктивных плоскостей.
4. Вычисление виртуальной точки как пересечения плоскостей.
5. Перенос ковариаций параметров плоскостей на координаты точки.
6. Статистическое сравнение одноимённых виртуальных точек двух эпох.
"""

from app.base import NamedPoint, Plane, Point
from app.batch import (
    ExtractionConfig,
    PointPairComparisonRunner,
    ReferencePoint,
    ReferencePointProcessingResult,
    ReferencePointReader,
    ScanNeighborhoodExtractor,
    SingleScanPointExtractor,
    SingleScanPointResult,
    ensure_unique_names,
    load_scan_from_file,
)
from app.cross_points import (
    CrossPoint,
    CrossPointExacter,
    PlaneGeometryDiagnostics,
    PlaneGeometryStatus,
)
from app.deformation import (
    DeformationAnalyzer,
    DeformationResult,
)
from app.scan import Scan, ScanPlane, ScanPoint

__all__ = [
    "Point",
    "NamedPoint",
    "Plane",
    "Scan",
    "ScanPoint",
    "ScanPlane",
    "CrossPoint",
    "CrossPointExacter",
    "PlaneGeometryDiagnostics",
    "PlaneGeometryStatus",
    "DeformationAnalyzer",
    "DeformationResult",
    "ExtractionConfig",
    "ReferencePoint",
    "ReferencePointReader",
    "ReferencePointProcessingResult",
    "ScanNeighborhoodExtractor",
    "SingleScanPointExtractor",
    "SingleScanPointResult",
    "PointPairComparisonRunner",
    "load_scan_from_file",
    "ensure_unique_names",
]
