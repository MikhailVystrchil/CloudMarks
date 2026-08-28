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
    PointPairComparisonRunner,
    ReferencePoint,
    ReferencePointProcessingResult,
    ReferencePointReader,
    ScanNeighborhoodExtractor,
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
    "ReferencePoint",
    "ReferencePointReader",
    "ReferencePointProcessingResult",
    "ScanNeighborhoodExtractor",
    "PointPairComparisonRunner",
]
