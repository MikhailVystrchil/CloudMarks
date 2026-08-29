import numpy as np
import pytest

from app.scan.Scan import Scan
from app.scan.ScanPoint import ScanPoint
from app.base.Point import Point


def test_add_point_updates_borders():
    scan = Scan("s")
    scan.add_point(Point(1.0, 2.0, 3.0))
    scan.add_point(Point(-1.0, 5.0, 0.0))

    assert scan.borders["x_min"] == -1.0
    assert scan.borders["x_max"] == 1.0
    assert scan.borders["y_min"] == 2.0
    assert scan.borders["y_max"] == 5.0
    assert scan.borders["z_min"] == 0.0
    assert scan.borders["z_max"] == 3.0


def test_add_point_rejects_wrong_type():
    scan = Scan("s")
    with pytest.raises(TypeError):
        scan.add_point(object())


def test_to_numpy_shape_and_values(make_scan):
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    scan = make_scan("s", coords)
    result = scan.to_numpy()

    assert result.shape == (2, 3)
    assert np.allclose(result, coords)


def test_to_numpy_empty_scan_has_correct_shape():
    scan = Scan("empty")
    result = scan.to_numpy()
    assert result.shape == (0, 3)


def test_subset_returns_independent_copy(make_scan):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    scan = make_scan("s", coords)

    subset = scan.subset(indices=np.array([0, 2]), copy_points=True)

    assert len(subset) == 2
    subset[0].x = 999.0
    assert scan[0].x == 0.0  # исходный Scan не должен измениться


def test_subset_rejects_out_of_range_indices(make_scan):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    scan = make_scan("s", coords)

    with pytest.raises(IndexError):
        scan.subset(indices=np.array([0, 5]))


def test_from_points_copies_scan_point_when_requested():
    original = ScanPoint(x=1.0, y=2.0, z=3.0)
    scan = Scan.from_points("s", [original], copy_points=True)

    scan[0].x = 42.0
    assert original.x == 1.0


def test_len_and_iter_are_consistent(make_scan):
    coords = np.zeros((5, 3))
    scan = make_scan("s", coords)

    assert len(scan) == 5
    assert len(list(iter(scan))) == 5
