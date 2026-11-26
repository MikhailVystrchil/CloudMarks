from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app import (
    CrossPointExacter,
    ExtractionConfig,
    PointPairComparisonRunner,
    SingleScanPointExtractor,
)


PARAMS = [
    ("default_radius", "Радиус окрестности, м", 0.60, float),
    ("min_neighborhood_points", "Минимум точек в окрестности", 600, int),
    ("min_points_per_plane", "Минимум точек на плоскость", 200, int),
    ("max_reference_distance_factor", "Допуск от базовой точки, ×R", 1.25, float),
    ("normal_k", "Соседей для расчёта нормалей", 8, int),
    ("cluster_eps", "DBSCAN: eps", 0.08, float),
    ("cluster_min_samples", "DBSCAN: min_samples", 3, int),
]


@dataclass
class Result:
    table: pd.DataFrame
    objects: dict = field(default_factory=dict)


def calculate(mode, paths, params, alpha, max_pair):
    scan1, scan2, references = paths

    if mode == "Фрагмент":
        exacter = CrossPointExacter(
            file_path=scan1,
            show_scans=False,
            normal_k=params["normal_k"],
            min_points_per_plane=params["min_points_per_plane"],
            cluster_eps=params["cluster_eps"],
            cluster_min_samples=params["cluster_min_samples"],
        )
        point = exacter.calculate_intersect_point()
        point.name = Path(scan1).stem
        return Result(point.to_dataframe(), {point.name: [point]})

    config = ExtractionConfig(**params)

    if mode == "Один скан":
        extractor = SingleScanPointExtractor.from_files(
            scan_path=scan1,
            reference_points_path=references,
            config=config,
            show_progress=False,
        )
        extractor.run(show_progress=False)
        objects = {r.name: [r.point] for r in extractor.results}
        return Result(extractor.to_dataframe(), objects)

    runner = PointPairComparisonRunner.from_files(
        epoch1_path=scan1,
        epoch2_path=scan2,
        config=config,
        alpha=alpha,
        max_pair_distance=max_pair,
    )
    table = runner.run_from_reference_file(
        reference_points_path=references,
        show_progress=False,
    )
    objects = {
        r.name: [r.point_epoch1, r.point_epoch2]
        for r in runner.processing_results
    }
    return Result(table, objects)
