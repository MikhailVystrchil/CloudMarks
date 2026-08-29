from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib import pyplot as plt

from app.scan.plotters.ScanPlotterABC import ScanPlotterABC


class ScanPlotterMPL(ScanPlotterABC):
    """
    Визуализирует Scan средствами matplotlib 3D.

    Равный масштаб координат устанавливается вручную, поскольку
    ``plt.axis("equal")`` не обеспечивает корректной геометрии
    для ``Axes3D``.
    """

    def __init__(
        self,
        fig_ax=None,
        is_show: bool = True,
        point_size: float = 1.0,
    ) -> None:
        self.fig, self.ax = (
            fig_ax
            if fig_ax is not None
            else (None, None)
        )
        self.is_show = bool(is_show)
        self.point_size = float(point_size)

        if self.point_size <= 0.0:
            raise ValueError(
                "point_size должен быть положительным."
            )

    def plot(self, scan):
        """
        Строит 3D-диаграмму облака точек.

        Returns
        -------
        tuple
            ``(figure, axes)`` matplotlib.
        """
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(
                projection="3d"
            )

        coordinates = scan.to_numpy()

        if len(coordinates) == 0:
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            if self.is_show:
                plt.show()

            return self.fig, self.ax

        colors = np.asarray(
            [
                np.asarray(point.color, dtype=np.float64)
                / 255.0
                for point in scan
            ],
            dtype=np.float64,
        )

        self.ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=colors,
            s=self.point_size,
        )

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self._set_equal_aspect_3d(
            coordinates=coordinates
        )

        if self.is_show:
            plt.show()

        return self.fig, self.ax

    def _set_equal_aspect_3d(
        self,
        coordinates: np.ndarray,
    ) -> None:
        """
        Задаёт равный масштаб трёх осей 3D-диаграммы.
        """
        minimums = coordinates.min(axis=0)
        maximums = coordinates.max(axis=0)

        center = (minimums + maximums) / 2.0
        half_ranges = (maximums - minimums) / 2.0
        half_range = max(float(half_ranges.max()), 1e-12)

        self.ax.set_xlim(
            center[0] - half_range,
            center[0] + half_range,
        )
        self.ax.set_ylim(
            center[1] - half_range,
            center[1] + half_range,
        )
        self.ax.set_zlim(
            center[2] - half_range,
            center[2] + half_range,
        )

        if hasattr(self.ax, "set_box_aspect"):
            self.ax.set_box_aspect(
                (1.0, 1.0, 1.0)
            )
