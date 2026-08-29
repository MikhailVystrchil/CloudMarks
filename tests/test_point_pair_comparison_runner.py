from __future__ import annotations

import numpy as np
import pytest

from app.batch.ExtractionConfig import ExtractionConfig
from app.batch.PointPairComparisonRunner import (
    PointPairComparisonRunner,
)
from app.batch.ReferencePoint import ReferencePoint
from app.scan.Scan import Scan
from app.scan.ScanPoint import ScanPoint


def _add_plane_points(
    scan: Scan,
    normal: tuple[float, float, float],
    center: tuple[float, float, float],
    extent: float,
    count: int,
    rng: np.random.Generator,
    noise: float = 0.0003,
) -> None:
    """
    Добавляет в Scan случайные точки одной плоскости.
    """
    normal_array = np.asarray(
        normal,
        dtype=np.float64,
    )
    normal_array /= np.linalg.norm(normal_array)

    helper = np.asarray(
        [1.0, 0.0, 0.0],
        dtype=np.float64,
    )

    if abs(normal_array[0]) > 0.9:
        helper = np.asarray(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )

    axis_a = (
        helper
        - helper.dot(normal_array) * normal_array
    )
    axis_a /= np.linalg.norm(axis_a)

    axis_b = np.cross(
        normal_array,
        axis_a,
    )

    u_coord = rng.uniform(
        -extent,
        extent,
        size=count,
    )
    v_coord = rng.uniform(
        -extent,
        extent,
        size=count,
    )

    coordinates = (
        np.asarray(center, dtype=np.float64)
        + u_coord[:, None] * axis_a
        + v_coord[:, None] * axis_b
        + rng.normal(
            scale=noise,
            size=(count, 3),
        )
    )

    for x_coord, y_coord, z_coord in coordinates:
        scan.add_point(
            ScanPoint(
                x=float(x_coord),
                y=float(y_coord),
                z=float(z_coord),
            )
        )


def _corner_scan(
    make_scan,
    center: tuple[float, float, float],
    seed: int,
) -> Scan:
    """
    Создаёт синтетический угол из трёх взаимно перпендикулярных плоскостей:

    - x = center_x;
    - y = center_y;
    - z = center_z.
    """
    random_generator = np.random.default_rng(seed)

    scan = make_scan(
        "corner",
        np.empty((0, 3)),
    )

    for normal in (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ):
        _add_plane_points(
            scan=scan,
            normal=normal,
            center=center,
            extent=0.5,
            count=60,
            rng=random_generator,
        )

    return scan


def _copy_scan(
    source_scan: Scan,
    *,
    scan_name: str,
) -> Scan:
    """
    Возвращает независимую копию Scan.

    Для test_runner_detects_no_displacement_for_identical_epochs
    важно обеспечить идентичные координаты обеих эпох, а не две
    независимые случайные реализации одной геометрии.
    """
    return Scan.from_points(
        scan_name=scan_name,
        points=list(source_scan),
        copy_points=True,
        include_normals=True,
        include_labels=True,
    )


def _make_runner(
    scan_epoch1: Scan,
    scan_epoch2: Scan,
    *,
    max_pair_distance: float | None = None,
) -> PointPairComparisonRunner:
    """
    Создаёт runner с параметрами, достаточными для синтетического угла.
    """
    return PointPairComparisonRunner(
        scan_epoch1=scan_epoch1,
        scan_epoch2=scan_epoch2,
        config=ExtractionConfig(
            default_radius=0.6,
            min_neighborhood_points=30,
            min_points_per_plane=10,
            max_reference_distance_factor=1.25,
            normal_k=12,
            cluster_eps=0.08,
            cluster_min_samples=3,
        ),
        max_pair_distance=max_pair_distance,
    )


def test_runner_detects_no_displacement_for_identical_epochs(
    make_scan,
):
    """
    Полностью идентичные облака должны дать SUCCESS и практически
    нулевое смещение.

    Этот тест намеренно не использует разные random seed для двух эпох:
    разные случайные реализации поверхности проверяют устойчивость
    сегментации, а не корректность PointPairComparisonRunner.
    """
    scan_epoch1 = _corner_scan(
        make_scan=make_scan,
        center=(1.0, 2.0, 3.0),
        seed=10,
    )

    scan_epoch2 = _copy_scan(
        source_scan=scan_epoch1,
        scan_name="corner_epoch2",
    )

    runner = _make_runner(
        scan_epoch1=scan_epoch1,
        scan_epoch2=scan_epoch2,
    )

    reference_points = [
        ReferencePoint(
            name="P1",
            x=1.0,
            y=2.0,
            z=3.0,
            radius=0.6,
        ),
    ]

    dataframe = runner.run(
        reference_points=reference_points,
        show_progress=False,
    )

    assert len(dataframe) == 1

    row = dataframe.iloc[0]

    assert row["processing_status"] == (
        PointPairComparisonRunner.SUCCESS
    )
    assert row["processing_message"] == "OK"

    assert np.isclose(
        row["pair_distance"],
        0.0,
        atol=1e-12,
    )

    assert np.isclose(
        row["displacement"],
        0.0,
        atol=1e-12,
    )

    assert np.isclose(
        row["displacement_mm"],
        0.0,
        atol=1e-9,
    )

    assert bool(row["analysis_reliable"])
    assert not bool(row["significant_t"])

    assert row["chi2_value"] is not None
    assert np.isclose(
        row["chi2_value"],
        0.0,
        atol=1e-12,
    )

    assert not bool(row["significant_chi2"])


def test_runner_marks_pair_unreliable_when_pair_distance_exceeds_limit(
    make_scan,
):
    """
    Контроль max_pair_distance должен отклонять пару, если её
    межэпоховое расстояние превышает заданный предел.

    Метод _combine_epoch_results не тестируется напрямую: проверяется
    публичное поведение полного runner на двух заметно смещённых углах.
    """
    scan_epoch1 = _corner_scan(
        make_scan=make_scan,
        center=(1.0, 2.0, 3.0),
        seed=20,
    )

    scan_epoch2 = _corner_scan(
        make_scan=make_scan,
        center=(1.0, 2.0, 3.5),
        seed=20,
    )

    runner = _make_runner(
        scan_epoch1=scan_epoch1,
        scan_epoch2=scan_epoch2,
        max_pair_distance=0.05,
    )

    reference_points = [
        ReferencePoint(
            name="P1",
            x=1.0,
            y=2.0,
            z=3.0,
            radius=0.6,
        ),
    ]

    dataframe = runner.run(
        reference_points=reference_points,
        show_progress=False,
    )

    assert len(dataframe) == 1

    row = dataframe.iloc[0]

    assert row["processing_status"] == (
        PointPairComparisonRunner.UNRELIABLE
    )

    assert row["pair_distance"] > 0.05

    assert "контрольный предел" in row[
        "processing_message"
    ]


def test_runner_rejects_duplicate_reference_names(
    make_scan,
):
    scan_epoch1 = _corner_scan(
        make_scan=make_scan,
        center=(0.0, 0.0, 0.0),
        seed=1,
    )

    scan_epoch2 = _copy_scan(
        source_scan=scan_epoch1,
        scan_name="corner_epoch2",
    )

    runner = _make_runner(
        scan_epoch1=scan_epoch1,
        scan_epoch2=scan_epoch2,
    )

    reference_points = [
        ReferencePoint(
            name="P1",
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        ReferencePoint(
            name="P1",
            x=0.0,
            y=0.0,
            z=0.0,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="уникальными",
    ):
        runner.run(
            reference_points=reference_points,
            show_progress=False,
        )


def test_runner_to_dataframe_before_run_is_empty(
    make_scan,
):
    scan_epoch1 = _corner_scan(
        make_scan=make_scan,
        center=(0.0, 0.0, 0.0),
        seed=5,
    )

    scan_epoch2 = _copy_scan(
        source_scan=scan_epoch1,
        scan_name="corner_epoch2",
    )

    runner = _make_runner(
        scan_epoch1=scan_epoch1,
        scan_epoch2=scan_epoch2,
    )

    dataframe = runner.to_dataframe()

    assert dataframe.empty
