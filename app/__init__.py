"""
CloudMarks.

Инструменты для безмарочного деформационного мониторинга сооружений
по данным наземного лазерного сканирования.

Основной алгоритм:
1. Извлечение локальной окрестности вокруг опорной точки.
2. Выделение трёх локальных плоскостей.
3. Вычисление виртуальной точки как пересечения плоскостей.
4. Оценка ковариации виртуальной точки.
5. Сравнение одноимённых виртуальных точек между двумя эпохами.
"""

from app.base import Plane, Point
from app.batch import (
    PointPairComparisonRunner,
    ReferencePoint,
    ReferencePointReader,
    ScanNeighborhoodExtractor,
)
from app.cross_points import CrossPoint, CrossPointExacter
from app.deformation import DeformationAnalyzer, DeformationResult
from app.scan import Scan, ScanPoint, ScanPlane

__all__ = [
    "Point",
    "Plane",
    "Scan",
    "ScanPoint",
    "ScanPlane",
    "CrossPoint",
    "CrossPointExacter",
    "DeformationAnalyzer",
    "DeformationResult",
    "ReferencePoint",
    "ReferencePointReader",
    "ScanNeighborhoodExtractor",
    "PointPairComparisonRunner",
]


