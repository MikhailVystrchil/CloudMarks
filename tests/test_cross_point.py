import numpy as np
import pytest

from app.cross_points.CrossPoint import CrossPoint


def test_load_covariance_computes_sigma_and_ellipsoid():
    point = CrossPoint(name="P1", x=1.0, y=2.0, z=3.0)
    covariance = np.diag([4e-6, 9e-6, 1e-6])

    point.load_covariance(covariance, confidence=0.95)

    assert point.reliable_accuracy
    assert np.allclose(point.sigma_xyz, [2e-3, 3e-3, 1e-3])
    assert point.ellipsoid is not None
    assert point.ellipsoid["semi_axes"].shape == (3,)


def test_load_covariance_rejects_non_symmetric_shape():
    point = CrossPoint(name="P1", x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValueError):
        point.load_covariance(np.eye(4))


def test_load_covariance_rejects_non_positive_semidefinite():
    point = CrossPoint(name="P1", x=0.0, y=0.0, z=0.0)
    bad_covariance = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError):
        point.load_covariance(bad_covariance)


def test_mark_unreliable_clears_accuracy_fields():
    point = CrossPoint(name="P1", x=0.0, y=0.0, z=0.0)
    point.load_covariance(np.eye(3) * 1e-6)

    point.mark_unreliable_accuracy()

    assert not point.reliable_accuracy
    assert point.sigma_xyz is None
    assert point.cov_xyz is None
    assert point.ellipsoid is None


def test_as_dict_reports_none_sigma_when_unreliable():
    point = CrossPoint(name="P1", x=0.0, y=0.0, z=0.0)
    point.mark_unreliable_accuracy()

    result = point.as_dict()
    assert result["sigma_x"] is None
    assert result["reliable_accuracy"] is False
