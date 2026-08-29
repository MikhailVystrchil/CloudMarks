from pathlib import Path

import pytest

from app.batch.ReferencePointReader import ReferencePointReader


def test_reads_csv_with_header(tmp_path: Path):
    file_path = tmp_path / "points.csv"
    file_path.write_text(
        "name,x,y,z,radius\nP1,1.0,2.0,3.0,0.5\nP2,4.0,5.0,6.0,\n",
        encoding="utf-8",
    )

    points = ReferencePointReader.read(file_path)

    assert len(points) == 2
    assert points[0].name == "P1"
    assert points[0].radius == 0.5
    assert points[1].radius is None


def test_reads_plain_txt_without_header(tmp_path: Path):
    file_path = tmp_path / "points.txt"
    file_path.write_text("P1 1.0 2.0 3.0\nP2 4.0 5.0 6.0 0.25\n", encoding="utf-8")

    points = ReferencePointReader.read(file_path)

    assert len(points) == 2
    assert points[0].radius is None
    assert points[1].radius == 0.25


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ReferencePointReader.read("does_not_exist.csv")


def test_empty_file_raises(tmp_path: Path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        ReferencePointReader.read(file_path)


def test_duplicate_names_raise(tmp_path: Path):
    file_path = tmp_path / "dup.csv"
    file_path.write_text(
        "name,x,y,z\nP1,1.0,2.0,3.0\nP1,4.0,5.0,6.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="повторяется"):
        ReferencePointReader.read(file_path)


def test_non_finite_coordinates_raise(tmp_path: Path):
    file_path = tmp_path / "bad.csv"
    file_path.write_text(
        "name,x,y,z\nP1,nan,2.0,3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ReferencePointReader.read(file_path)


def test_comment_and_blank_lines_are_skipped(tmp_path: Path):
    file_path = tmp_path / "commented.txt"
    file_path.write_text(
        "# comment\n\nP1 1.0 2.0 3.0\n# another comment\nP2 4.0 5.0 6.0\n",
        encoding="utf-8",
    )

    points = ReferencePointReader.read(file_path)
    assert [p.name for p in points] == ["P1", "P2"]
