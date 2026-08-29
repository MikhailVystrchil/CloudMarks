from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def ensure_unique_names(
    names: Iterable[str],
    *,
    entity_name: str = "объектов",
) -> None:
    """
    Проверяет уникальность строковых идентификаторов.

    Parameters
    ----------
    names:
        Последовательность идентификаторов.
    entity_name:
        Наименование сущностей для текста исключения.

    Raises
    ------
    ValueError
        Если среди переданных имён обнаружены повторы.
    """
    normalized_names = [str(name) for name in names]
    duplicates = sorted(
        name
        for name, count in Counter(normalized_names).items()
        if count > 1
    )

    if duplicates:
        raise ValueError(
            f"Имена {entity_name} должны быть уникальными. "
            f"Повторяются: {duplicates}."
        )
