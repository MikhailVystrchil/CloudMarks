import numpy as np
import pytest

from app.batch.ReferencePoint import ReferencePoint
from app.batch.ScanNeighborhoodExtractor import ScanNeighborhoodExtractor


def test_extract_sphere_returns_points_within_radius(make_scan):
    grid_x, grid_y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    coords = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]
    ).astype(float)
    scan = make_scan("grid", coords)

    extractor = ScanNeighborhoodExtractor(scan)
    reference = ReferencePoint(name="P1", x=0.0, y=0.0, z=0.0)

    neighborhood = extractor.extract_sphere(reference, radius=1.5)

    xyz = neighborhood.to_numpy()
    distances = np.linalg.norm(xyz, axis=1)
    assert len(neighborhood) > 0
    assert np.all(distances <= 1.5 + 1e-9)


def test_extract_spheres_rejects_duplicate_names(make_scan):
    scan = make_scan("s", np.zeros((5, 3)))
    extractor = ScanNeighborhoodExtractor(scan)

    points = [
        ReferencePoint(name="P1", x=0.0, y=0.0, z=0.0),
        ReferencePoint(name="P1", x=1.0, y=1.0, z=1.0),
    ]

    with pytest.raises(ValueError):
        extractor.extract_spheres(points, default_radius=1.0)


def test_constructor_rejects_empty_scan():
    from app.scan.Scan import Scan

    with pytest.raises(ValueError):
        ScanNeighborhoodExtractor(Scan("empty"))


def test_query_indices_many_rejects_mismatched_radius_length(make_scan):
    scan = make_scan("s", np.random.default_rng(0).normal(size=(20, 3)))
    extractor = ScanNeighborhoodExtractor(scan)

    with pytest.raises(ValueError):
        extractor.query_indices_many(
            centers=np.zeros((3, 3)),
            radius=np.array([1.0, 1.0]),  # длина не совпадает с centers
        )


def test_extract_by_indices_rejects_out_of_range(make_scan):
    scan = make_scan("s", np.zeros((3, 3)))
    extractor = ScanNeighborhoodExtractor(scan)

    with pytest.raises(IndexError):
        extractor.extract_by_indices(
            point_indices=np.array([0, 10]),
            reference_name="P1",
            radius=1.0,
        )
