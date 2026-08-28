from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from app.batch.ReferencePoint import ReferencePoint


class ReferencePointReader:
    """
    Читает опорные точки из CSV, TSV или TXT.

    Поддерживаемые форматы
    ----------------------

    1. Табличный формат с заголовком:

        name,x,y,z
        P001,126.4059,119.8113,18.9068
        P002,126.3835,116.7849,18.8980

    или:

        name;x;y;z;radius
        P001;126.4059;119.8113;18.9068;0.25

    2. Простой пробельно-разделённый TXT без заголовка:

        A_20_5_r 126.4059 119.8113 18.9068
        A_19_5_r 126.3835 116.7849 18.8980
        A_19_6_r 126.3986 116.7573 22.4936

    В текстовом формате допускается пятое значение — индивидуальный радиус:

        A_20_5_r 126.4059 119.8113 18.9068 0.25

    Пустые строки и строки, начинающиеся с '#', игнорируются.
    """

    REQUIRED_COLUMNS = ("name", "x", "y", "z")
    OPTIONAL_COLUMNS = ("radius",)

    @classmethod
    def read(
        cls,
        file_path: str | Path,
    ) -> list[ReferencePoint]:
        """
        Считывает файл и возвращает список уникальных ReferencePoint.
        """
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Файл опорных точек не найден: {file_path}"
            )

        logger.info("Чтение опорных точек: {}", file_path)

        dataframe, format_name = cls._read_dataframe(file_path)

        logger.info(
            "Распознан формат файла опорных точек: {}",
            format_name,
        )

        if dataframe.empty:
            raise ValueError(
                f"Файл опорных точек пуст: {file_path}"
            )

        points: list[ReferencePoint] = []
        seen_names: set[str] = set()

        for dataframe_index, row in dataframe.iterrows():
            row_number = int(dataframe_index) + 2

            point = cls._row_to_reference_point(
                row=row,
                row_number=row_number,
                source_path=file_path,
            )

            if point.name in seen_names:
                raise ValueError(
                    f"Имя опорной точки '{point.name}' повторяется "
                    f"в файле '{file_path}'. Имена должны быть уникальными."
                )

            seen_names.add(point.name)
            points.append(point)

        logger.success(
            "Загружено опорных точек: {}",
            len(points),
        )

        return points

    @classmethod
    def _read_dataframe(
        cls,
        file_path: Path,
    ) -> tuple[pd.DataFrame, str]:
        """
        Определяет наличие заголовка и читает файл в DataFrame с колонками:
        name, x, y, z, [radius].
        """
        first_data_line = cls._find_first_data_line(file_path)

        if first_data_line is None:
            raise ValueError(
                f"Файл опорных точек не содержит данных: {file_path}"
            )

        normalized_tokens = cls._split_and_normalize(first_data_line)

        if cls._is_header(normalized_tokens):
            dataframe = cls._read_header_table(file_path)
            return dataframe, "таблица с заголовком"

        dataframe = cls._read_plain_text_table(file_path)
        return dataframe, "пробельно-разделённый TXT без заголовка"

    @staticmethod
    def _find_first_data_line(
        file_path: Path,
    ) -> str | None:
        """
        Находит первую непустую и некомментированную строку.
        """
        try:
            with file_path.open(
                "r",
                encoding="utf-8-sig",
            ) as input_file:
                for raw_line in input_file:
                    line = raw_line.strip()

                    if not line or line.startswith("#"):
                        continue

                    return line

        except UnicodeDecodeError as error:
            raise ValueError(
                f"Не удалось прочитать файл '{file_path}' как UTF-8: {error}"
            ) from error

        return None

    @staticmethod
    def _split_and_normalize(
        line: str,
    ) -> list[str]:
        """
        Делит строку по запятой, точке с запятой, табуляции или пробелам.

        Пример:
            'name,x,y,z' -> ['name', 'x', 'y', 'z']
            'A_20_5_r 126.4059 119.8113 18.9068'
            -> ['a_20_5_r', '126.4059', '119.8113', '18.9068']
        """
        normalized_line = (
            line.replace(";", " ")
            .replace(",", " ")
            .replace("\t", " ")
        )

        return [
            token.strip().lower()
            for token in normalized_line.split()
            if token.strip()
        ]

    @classmethod
    def _is_header(
        cls,
        tokens: list[str],
    ) -> bool:
        """
        Определяет заголовок строго по именам колонок.

        Только строка, содержащая name, x, y и z, считается заголовком.
        Строка вида:
            A_20_5_r 126.4059 119.8113 18.9068
        всегда будет обработана как данные.
        """
        required_columns = {"name", "x", "y", "z"}
        normalized_tokens = set(tokens)

        return required_columns.issubset(normalized_tokens)

    @classmethod
    def _is_header(
        cls,
        tokens: list[str],
    ) -> bool:
        """
        Возвращает True только если первая строка действительно является
        заголовком с полями name, x, y, z.

        Обычные имена точек, например A_20_5_r, CP_0_1, B_5_2_l,
        не могут быть ошибочно распознаны как заголовок.
        """
        normalized = {
            token.strip().lower()
            for token in tokens
            if token.strip()
        }

        required = {"name", "x", "y", "z"}

        return required.issubset(normalized)

    @classmethod
    def _read_header_table(
        cls,
        file_path: Path,
    ) -> pd.DataFrame:
        """
        Читает CSV/TSV/semicolon-таблицу с заголовком.

        sep=None с python-engine автоматически определяет запятую, точку
        с запятой или табуляцию. Для пробельно-разделённого заголовка
        используется резервный режим.
        """
        try:
            dataframe = pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                comment="#",
                skipinitialspace=True,
                encoding="utf-8-sig",
            )
        except Exception as error:
            raise ValueError(
                f"Не удалось прочитать таблицу опорных точек "
                f"'{file_path}': {error}"
            ) from error

        dataframe.columns = [
            str(column).strip().lower()
            for column in dataframe.columns
        ]

        missing_columns = (
            set(cls.REQUIRED_COLUMNS)
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "В таблице опорных точек отсутствуют обязательные колонки: "
                f"{sorted(missing_columns)}. "
                "Ожидаются: name,x,y,z[,radius]."
            )

        return dataframe

    @staticmethod
    def _read_plain_text_table(
            file_path: Path,
    ) -> pd.DataFrame:
        """
        Читает TXT без заголовка вида:

            name X Y Z
        или:
            name X Y Z radius
        """
        try:
            raw_dataframe = pd.read_csv(
                file_path,
                sep=r"\s+",
                engine="python",
                comment="#",
                header=None,
                encoding="utf-8-sig",
            )
        except Exception as error:
            raise ValueError(
                f"Не удалось прочитать TXT-файл опорных точек "
                f"'{file_path}': {error}"
            ) from error

        column_count = raw_dataframe.shape[1]

        if column_count not in {4, 5}:
            raise ValueError(
                "TXT-файл опорных точек должен содержать 4 или 5 полей в строке: "
                "name X Y Z [radius]. "
                f"Фактически найдено столбцов: {column_count}."
            )

        raw_dataframe.columns = (
            ["name", "x", "y", "z"]
            if column_count == 4
            else ["name", "x", "y", "z", "radius"]
        )

        return raw_dataframe

    @staticmethod
    def _row_to_reference_point(
        row: pd.Series,
        row_number: int,
        source_path: Path,
    ) -> ReferencePoint:
        """
        Преобразует строку DataFrame в валидированный ReferencePoint.
        """
        try:
            name = str(row["name"]).strip()

            if not name or name.lower() == "nan":
                raise ValueError("пустое имя точки")

            x_coord = float(row["x"])
            y_coord = float(row["y"])
            z_coord = float(row["z"])

            radius: float | None = None

            if "radius" in row.index and pd.notna(row["radius"]):
                radius = float(row["radius"])

                if radius <= 0:
                    raise ValueError(
                        f"радиус должен быть положительным, получено {radius}"
                    )

            return ReferencePoint(
                name=name,
                x=x_coord,
                y=y_coord,
                z=z_coord,
                radius=radius,
            )

        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Ошибка в строке {row_number} файла '{source_path}': "
                f"{error}"
            ) from error
