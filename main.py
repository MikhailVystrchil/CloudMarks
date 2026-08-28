"""
Демонстрационный сценарий метода: извлечение виртуальной точки пересечения
трёх плоскостей из локальных фрагментов облака точек для двух эпох съёмки
и статистическая проверка значимости обнаруженного смещения.
"""

from app.cross_points.CrossPointExacter import CrossPointExacter
from app.deformation.DeformationAnalyzer import DeformationAnalyzer


def extract_virtual_point(file_path: str, name: str, labels=None):
    cpe = CrossPointExacter(file_path, labels=labels, show_scans=False)
    cpe.calculate_planes()
    point = cpe.calculate_intersect_point()
    point.name = name
    print(cpe.get_result_str())
    return point


if __name__ == "__main__":
    point_epoch1 = extract_virtual_point("data/1_A_10_2_l.txt", name="P1")
    point_epoch2 = extract_virtual_point("data/2_A_10_2_l.txt", name="P1")

    analyzer = DeformationAnalyzer(alpha=0.05)
    analyzer.analyze_point_sets([point_epoch1], [point_epoch2])
    analyzer.print_summary()