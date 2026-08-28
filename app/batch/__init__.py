"""
Пакетная обработка пары крупных разновременных облаков точек.

Компоненты пакета позволяют:
- считывать список опорных точек;
- выделять локальные сферические окрестности через cKDTree;
- получать виртуальные точки в обеих эпохах;
- контролировать качество геометрии;
- рассчитывать смещения и экспортировать результаты.
"""

from app.batch.PointPairComparisonRunner import (
    PointPairComparisonRunner,
    ReferencePointProcessingResult,
)
from app.batch.ReferencePoint import ReferencePoint
from app.batch.ReferencePointReader import ReferencePointReader
from app.batch.ScanNeighborhoodExtractor import (
    ScanNeighborhoodExtractor,
)

__all__ = [
    "ReferencePoint",
    "ReferencePointReader",
    "ScanNeighborhoodExtractor",
    "ReferencePointProcessingResult",
    "PointPairComparisonRunner",
]
