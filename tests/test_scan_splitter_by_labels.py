import numpy as np

from app.scan.utils.ScanSplitterByLabels import ScanSplitterByLabels


def test_split_groups_points_by_label(make_scan):
    scan = make_scan("s", np.zeros((6, 3)))

    for index, point in enumerate(scan):
        point.labels = index % 3

    splitter = ScanSplitterByLabels(scan)
    groups = splitter.split()

    assert set(groups) == {0, 1, 2}
    assert all(len(group) == 2 for group in groups.values())


def test_split_excludes_noise_by_default(make_scan):
    scan = make_scan("s", np.zeros((4, 3)))

    for index, point in enumerate(scan):
        point.labels = -1 if index == 0 else 0

    splitter = ScanSplitterByLabels(scan, include_noise=False)
    groups = splitter.split()

    assert -1 not in groups
    assert len(groups[0]) == 3


def test_split_includes_noise_when_requested(make_scan):
    scan = make_scan("s", np.zeros((4, 3)))

    for index, point in enumerate(scan):
        point.labels = -1 if index == 0 else 0

    splitter = ScanSplitterByLabels(scan, include_noise=True)
    groups = splitter.split()

    assert -1 in groups
    assert len(groups[-1]) == 1


def test_points_without_label_attribute_are_ignored(make_scan):
    scan = make_scan("s", np.zeros((2, 3)))
    # labels не назначены ни одной точке
    splitter = ScanSplitterByLabels(scan)
    groups = splitter.split()

    assert groups == {}
