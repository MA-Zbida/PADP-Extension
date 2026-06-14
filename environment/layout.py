"""Warehouse layout generation for the continuous collaborative carry task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from environment.geometry import Rect, vec


@dataclass
class WarehouseLayout:
    depot: Rect
    shipping: Rect
    racks: List[Rect]
    pick_faces: List[np.ndarray]
    spawn_points: List[np.ndarray]
    goal_points: List[np.ndarray]


def _grid_points(rect: Rect, rows: int, cols: int, margin: float = 0.04) -> List[np.ndarray]:
    xs = np.linspace(rect.x1 + margin, rect.x2 - margin, max(1, cols))
    ys = np.linspace(rect.y1 + margin, rect.y2 - margin, max(1, rows))
    return [vec(float(x), float(y)) for y in ys for x in xs]


def make_warehouse_layout(n_obstacles: int, max_obstacles: int) -> WarehouseLayout:
    depot = Rect(0.04, 0.74, 0.30, 0.96)
    shipping = Rect(0.72, 0.04, 0.96, 0.28)

    rack_candidates = [
        Rect(0.34, 0.20, 0.40, 0.42),
        Rect(0.52, 0.20, 0.58, 0.42),
        Rect(0.34, 0.56, 0.40, 0.78),
        Rect(0.52, 0.56, 0.58, 0.78),
        Rect(0.70, 0.38, 0.76, 0.60),
        Rect(0.18, 0.38, 0.24, 0.60),
    ]
    rack_count = min(max(n_obstacles, 0), max(max_obstacles, n_obstacles), len(rack_candidates))
    racks = rack_candidates[:rack_count]

    pick_faces: List[np.ndarray] = []
    for rack in racks:
        center = rack.center
        candidates = [
            vec(rack.x1 - 0.075, float(center[1])),
            vec(rack.x2 + 0.075, float(center[1])),
            vec(float(center[0]), rack.y1 - 0.075),
            vec(float(center[0]), rack.y2 + 0.075),
        ]
        for point in candidates:
            if 0.06 <= point[0] <= 0.94 and 0.06 <= point[1] <= 0.94:
                if not (depot.x1 <= point[0] <= depot.x2 and depot.y1 <= point[1] <= depot.y2):
                    if not (shipping.x1 <= point[0] <= shipping.x2 and shipping.y1 <= point[1] <= shipping.y2):
                        pick_faces.append(point)

    spawn_points = _grid_points(depot, rows=3, cols=4, margin=0.045)
    goal_points = _grid_points(shipping, rows=2, cols=3, margin=0.06)

    return WarehouseLayout(
        depot=depot,
        shipping=shipping,
        racks=racks,
        pick_faces=pick_faces,
        spawn_points=spawn_points,
        goal_points=goal_points,
    )
