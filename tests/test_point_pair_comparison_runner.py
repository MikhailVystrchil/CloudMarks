import numpy as np

from app.batch.PointPairComparisonRunner import PointPairComparisonRunner
from app.batch.ReferencePoint import ReferencePoint
from app.scan.ScanPoint import ScanPoint


def _corner_scan(make_scan, center, seed):
    rng = np.random.default_rng(seed)
    scan = make_scan("corner", np.empty((0, 3)))

    def add_plane(normal, extent=0.5, count=60, noise=0.0003):
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(normal[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis_a = helper - helper.dot(normal) * normal
        axis_a /= np.linalg.norm(axis_a)
        axis_b = np.cross(normal, axis_a)
        u = rng.uniform(-extent, extent, size=count)
        v = rng.uniform(-extent, extent, size=count)
        coords = (
            np.asarray(center)
            + u[:, None] * axis_a
            + v[:, None] * axis_b
            + rng.normal(scale=noise, size=(count, 3))
        )
        for xc, yc, zc in coords:
            scan.add_point(ScanPoint(x=float(xc), y=float(yc), z=float(zc)))

    add_plane((1, 0, 0))
    add_plane((0, 1, 0))
    add_plane((0, 0, 1))
    return scan


def test_runner_detects_no_displacement_for_identical_epochs(make_scan):
    scan1 = _corner_scan(make_scan, center=(1.0, 2.0, 3.0), seed=10)
    scan2 = _corner_scan(make_scan, center=(1.0, 2.0, 3.0), seed=11)

    runner = PointPairComparisonRunner(
        scan_epoch1=scan1,
        scan_epoch2=scan2,
        default_radius=0.6,
        min_neighborhood_points=30,
        min_points_per_plane=10,
    )

    reference_points = [
        ReferencePoint(name="P1", x=1.0, y=2.0, z=3.0, radius=0.6),
    ]

    dataframe = runner.run(reference_points, show_progress=False)

    assert len(dataframe) == 1
    row = dataframe.iloc[0]
    assert row["processing_status"] == PointPairComparisonRunner.SUCCESS
    assert abs(row["displacement_mm"]) < 5.0
    assert row["significant_t"] in (False, np.False_)


def test_runner_rejects_duplicate_reference_names(make_scan):
    import pytest

    scan1 = _corner_scan(make_scan, center=(0.0, 0.0, 0.0), seed=1)
    scan2 = _corner_scan(make_scan, center=(0.0, 0.0, 0.0), seed=2)

    runner = PointPairComparisonRunner(
        scan_epoch1=scan1,
        scan_epoch2=scan2,
        default_radius=0.6,
    )

    points = [
        ReferencePoint(name="P1", x=0.0, y=0.0, z=0.0),
        ReferencePoint(name="P1", x=0.0, y=0.0, z=0.0),
    ]

    with pytest.raises(ValueError):
        runner.run(points, show_progress=False)


def test_runner_to_dataframe_before_run_is_empty(make_scan):
    scan1 = _corner_scan(make_scan, center=(0.0, 0.0, 0.0), seed=5)
    scan2 = _corner_scan(make_scan, center=(0.0, 0.0, 0.0), seed=6)

    runner = PointPairComparisonRunner(
        scan_epoch1=scan1,
        scan_epoch2=scan2,
        default_radius=0.6,
    )

    assert runner.to_dataframe().empty
