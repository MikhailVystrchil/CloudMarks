from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from app.batch.ReferencePoint import ReferencePoint


class ReferencePointReader:
    """
    Читает опорные точки из CSV, TSV или TXT.

    Поддерживаются два варианта:

    1. Таблица с заголовком:
       ``name,x,y,z[,radius]``

    2. Текстовый файл без заголовка:
       ``name X Y Z [radius]``

    Пустые строки и строки с первым символом ``#`` игнорируются.
    """

    REQUIRED_COLUMNS = ("name", "x", "y", "z")
    OPTIONAL_COLUMNS = ("radius",)

    @classmethod
    def read(
        cls,
        file_path: str | Path,
    ) -> list[ReferencePoint]:
        """
        Загружает и валидирует уникальный набор опорных точек.
        """
        source_path = Path(file_path)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Файл опорных точек не найден: {source_path}"
            )

        logger.info(
            "Чтение опорных точек: {}",
            source_path,
        )

        dataframe, format_name = cls._read_dataframe(
            file_path=source_path,
        )

        logger.info(
            "Распознан формат файла опорных точек: {}",
            format_name,
        )

        if dataframe.empty:
            raise ValueError(
                f"Файл опорных точек пуст: {source_path}"
            )

        points: list[ReferencePoint] = []
        seen_names: set[str] = set()

        for dataframe_index, row in dataframe.iterrows():
            row_number = int(dataframe_index) + 2

            point = cls._row_to_reference_point(
                row=row,
                row_number=row_number,
                source_path=source_path,
            )

            if point.name in seen_names:
                raise ValueError(
                    f"Имя опорной точки '{point.name}' повторяется "
                    f"в файле '{source_path}'. "
                    "Имена должны быть уникальными."
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
        Определяет наличие заголовка и читает таблицу.
        """
        first_data_line = cls._find_first_data_line(
            file_path=file_path,
        )

        if first_data_line is None:
            raise ValueError(
                f"Файл опорных точек не содержит данных: {file_path}"
            )

        tokens = cls._split_and_normalize(
            line=first_data_line,
        )

        if cls._is_header(tokens=tokens):
            return (
                cls._read_header_table(file_path=file_path),
                "таблица с заголовком",
            )

        return (
            cls._read_plain_text_table(file_path=file_path),
            "пробельно-разделённый TXT без заголовка",
        )

    @staticmethod
    def _find_first_data_line(
        file_path: Path,
    ) -> str | None:
        """
        Возвращает первую строку, не являющуюся пустой или комментарием.
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
                f"Не удалось прочитать файл '{file_path}' как UTF-8: "
                f"{error}"
            ) from error

        return None

    @staticmethod
    def _split_and_normalize(
        line: str,
    ) -> list[str]:
        """
        Делит строку по `,`, `;`, табуляции и пробелам.
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
        Возвращает True, только если строка содержит обязательные имена полей.
        """
        required_columns = set(cls.REQUIRED_COLUMNS)
        normalized_tokens = {
            token.strip().lower()
            for token in tokens
            if token.strip()
        }

        return required_columns.issubset(
            normalized_tokens
        )

    @classmethod
    def _read_header_table(
        cls,
        file_path: Path,
    ) -> pd.DataFrame:
        """
        Читает CSV, TSV или таблицу с разделителем `;`.
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
                "Не удалось прочитать таблицу опорных точек "
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
        Читает TXT без заголовка:

        ``name X Y Z``
        или
        ``name X Y Z radius``.
        """
        try:
            dataframe = pd.read_csv(
                file_path,
                sep=r"\s+",
                engine="python",
                comment="#",
                header=None,
                encoding="utf-8-sig",
            )
        except Exception as error:
            raise ValueError(
                "Не удалось прочитать TXT-файл опорных точек "
                f"'{file_path}': {error}"
            ) from error

        column_count = dataframe.shape[1]

        if column_count not in {4, 5}:
            raise ValueError(
                "TXT-файл опорных точек должен содержать 4 или 5 полей: "
                "name X Y Z [radius]. "
                f"Фактически найдено столбцов: {column_count}."
            )

        dataframe.columns = (
            ["name", "x", "y", "z"]
            if column_count == 4
            else ["name", "x", "y", "z", "radius"]
        )

        return dataframe

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

            coordinates = np.asarray(
                [x_coord, y_coord, z_coord],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    "координаты содержат NaN или Inf"
                )

            radius: float | None = None

            if (
                "radius" in row.index
                and pd.notna(row["radius"])
            ):
                radius = float(row["radius"])

                if not np.isfinite(radius) or radius <= 0.0:
                    raise ValueError(
                        "радиус должен быть конечным "
                        f"положительным числом, получено {radius}"
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
                f"Ошибка в строке {row_number} файла "
                f"'{source_path}': {error}"
            ) from error
