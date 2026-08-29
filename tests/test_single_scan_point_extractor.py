from pathlib import Path

import numpy as np
import pytest

from app.batch.ExtractionConfig import ExtractionConfig
from app.batch.ReferencePoint import ReferencePoint
from app.batch.SingleScanPointExtractor import (
    SingleScanPointExtractor,
)
from app.scan.ScanPoint import ScanPoint


def _add_plane_points(
    scan,
    normal,
    center,
    extent,
    count,
    rng,
    noise=0.0003,
):
    """
    Добавляет в Scan случайные точки одной плоскости.
    """
    normal = np.asarray(
        normal,
        dtype=float,
    )
    normal /= np.linalg.norm(normal)

    helper = np.asarray(
        [1.0, 0.0, 0.0],
        dtype=float,
    )

    if abs(normal[0]) > 0.9:
        helper = np.asarray(
            [0.0, 1.0, 0.0],
            dtype=float,
        )

    axis_a = helper - helper.dot(normal) * normal
    axis_a /= np.linalg.norm(axis_a)

    axis_b = np.cross(normal, axis_a)

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
        np.asarray(center, dtype=float)
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


@pytest.fixture
def corner_scan(make_scan):
    """
    Синтетический угол из трёх взаимно перпендикулярных плоскостей:

        x = 1;
        y = 2;
        z = 3.

    Ожидаемая виртуальная точка: (1, 2, 3).
    """
    scan = make_scan(
        "corner_scan",
        np.empty((0, 3)),
    )

    random_generator = np.random.default_rng(42)
    center = (1.0, 2.0, 3.0)

    _add_plane_points(
        scan=scan,
        normal=(1.0, 0.0, 0.0),
        center=center,
        extent=0.5,
        count=50,
        rng=random_generator,
    )

    _add_plane_points(
        scan=scan,
        normal=(0.0, 1.0, 0.0),
        center=center,
        extent=0.5,
        count=50,
        rng=random_generator,
    )

    _add_plane_points(
        scan=scan,
        normal=(0.0, 0.0, 1.0),
        center=center,
        extent=0.5,
        count=50,
        rng=random_generator,
    )

    return scan


def test_extractor_recovers_known_corner_point(
    corner_scan,
):
    reference_points = [
        ReferencePoint(
            name="P1",
            x=1.0,
            y=2.0,
            z=3.0,
            radius=0.6,
        ),
    ]

    extractor = SingleScanPointExtractor(
        scan=corner_scan,
        config=ExtractionConfig(
            default_radius=0.6,
            min_neighborhood_points=30,
            min_points_per_plane=10,
        ),
    )

    extractor.run(
        reference_points=reference_points,
        show_progress=False,
    )

    assert len(extractor.results) == 1

    result = extractor.results[0]

    assert result.status == SingleScanPointExtractor.SUCCESS
    assert result.point is not None

    recovered = np.asarray(
        [
            result.point.x,
            result.point.y,
            result.point.z,
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        recovered,
        [1.0, 2.0, 3.0],
        atol=0.05,
    )


def test_extractor_reports_problem_for_small_neighborhood(
    corner_scan,
):
    reference_points = [
        ReferencePoint(
            name="P1",
            x=1.0,
            y=2.0,
            z=3.0,
            radius=0.05,
        ),
    ]

    extractor = SingleScanPointExtractor(
        scan=corner_scan,
        config=ExtractionConfig(
            default_radius=0.05,
            min_neighborhood_points=30,
            min_points_per_plane=10,
        ),
    )

    extractor.run(
        reference_points=reference_points,
        show_progress=False,
    )

    result = extractor.results[0]

    assert result.status != SingleScanPointExtractor.SUCCESS
    assert result.neighborhood_points is not None
    assert result.neighborhood_points < 30


def test_extractor_rejects_duplicate_reference_names(
    corner_scan,
):
    reference_points = [
        ReferencePoint(
            name="P1",
            x=1.0,
            y=2.0,
            z=3.0,
        ),
        ReferencePoint(
            name="P1",
            x=1.0,
            y=2.0,
            z=3.0,
        ),
    ]

    extractor = SingleScanPointExtractor(
        scan=corner_scan,
        config=ExtractionConfig(
            default_radius=0.6,
        ),
    )

    with pytest.raises(
        ValueError,
        match="уникальными",
    ):
        extractor.run(
            reference_points=reference_points,
            show_progress=False,
        )


def test_to_dataframe_and_export_csv(
    corner_scan,
    tmp_path: Path,
):
    reference_points = [
        ReferencePoint(
            name="P1",
            x=1.0,
            y=2.0,
            z=3.0,
            radius=0.6,
        ),
    ]

    extractor = SingleScanPointExtractor(
        scan=corner_scan,
        config=ExtractionConfig(
            default_radius=0.6,
            min_neighborhood_points=30,
            min_points_per_plane=10,
        ),
    )

    extractor.run(
        reference_points=reference_points,
        show_progress=False,
    )

    dataframe = extractor.to_dataframe()

    assert list(dataframe["name"]) == ["P1"]

    expected_columns = {
        "name",
        "reference_x",
        "reference_y",
        "reference_z",
        "radius",
        "neighborhood_points",
        "reference_distance",
        "x",
        "y",
        "z",
        "geometry_status",
        "reliable_accuracy",
        "sigma_x",
        "sigma_y",
        "sigma_z",
        "status",
        "message",
    }

    assert expected_columns.issubset(
        set(dataframe.columns)
    )

    output_path = tmp_path / "out" / "points.csv"

    returned_path = extractor.export_csv(
        output_path
    )

    assert returned_path == output_path
    assert output_path.is_file()
    assert output_path.read_text(
        encoding="utf-8"
    ).strip()


def test_run_without_reference_points_raises(
    corner_scan,
):
    extractor = SingleScanPointExtractor(
        scan=corner_scan,
        config=ExtractionConfig(
            default_radius=0.6,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Список опорных точек пуст",
    ):
        extractor.run(show_progress=False)
