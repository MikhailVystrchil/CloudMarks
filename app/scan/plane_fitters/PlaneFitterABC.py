from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.scan.Scan import Scan


class PlaneFitterABC(ABC):
    """
    Абстрактный интерфейс аппроксимации плоскости по Scan.
    """

    def __init__(
        self,
        scan: Scan,
    ) -> None:
        self.scan = scan

    def _scan_to_numpy(self) -> np.ndarray:
        """
        Координаты точек скана как массив формы ``(N, 3)``.
        """
        return self.scan.to_numpy()

    @abstractmethod
    def fit_plane(
        self,
        *args,
        **kwargs,
    ):
        """
        Аппроксимирует плоскость по ``self.scan``.
        """
        raise NotImplementedError
