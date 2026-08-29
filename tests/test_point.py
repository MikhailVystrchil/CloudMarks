import numpy as np
import pytest

from app.base.Point import Point, NamedPoint


def test_points_within_tolerance_are_equal():
    a = Point(1.0, 2.0, 3.0)
    b = Point(1.0 + 1e-6, 2.0 - 1e-6, 3.0)
    assert a == b


def test_points_outside_tolerance_are_not_equal():
    a = Point(1.0, 2.0, 3.0)
    b = Point(1.0 + 1e-3, 2.0, 3.0)
    assert a != b


def test_point_equality_rejects_other_types():
    assert Point(1.0, 2.0, 3.0).__eq__("not a point") is NotImplemented


def test_point_is_unhashable():
    with pytest.raises(TypeError):
        hash(Point(1.0, 2.0, 3.0))


def test_point_as_array_matches_coordinates():
    point = Point(1.5, -2.5, 3.5)
    assert np.array_equal(point.as_array(), np.array([1.5, -2.5, 3.5]))


def test_named_point_repr_contains_name():
    point = NamedPoint("P1", 1.0, 2.0, 3.0)
    assert "P1" in repr(point)
    assert "P1" in str(point)
