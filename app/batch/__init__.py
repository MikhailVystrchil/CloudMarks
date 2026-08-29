"""
Пакетная обработка облаков точек по опорным координатам.
"""

from app.batch.ExtractionConfig import ExtractionConfig
from app.batch.PointPairComparisonRunner import (
    PointPairComparisonRunner,
    ReferencePointProcessingResult,
)
from app.batch.ReferencePoint import ReferencePoint
from app.batch.ReferencePointReader import ReferencePointReader
from app.batch.ScanNeighborhoodExtractor import (
    ScanNeighborhoodExtractor,
)
from app.batch.SingleScanPointExtractor import (
    SingleScanPointExtractor,
    SingleScanPointResult,
)
from app.batch.scan_loading import load_scan_from_file
from app.batch.validation import ensure_unique_names

__all__ = [
    "ExtractionConfig",
    "PointPairComparisonRunner",
    "ReferencePointProcessingResult",
    "ReferencePoint",
    "ReferencePointReader",
    "ScanNeighborhoodExtractor",
    "SingleScanPointExtractor",
    "SingleScanPointResult",
    "load_scan_from_file",
    "ensure_unique_names",
]
