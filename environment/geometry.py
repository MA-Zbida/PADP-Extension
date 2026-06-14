"""Continuous 2D geometry helpers for warehouse MARL dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Vec2 = np.ndarray


@dataclass(frozen=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5], dtype=np.float32)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def vec(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float32)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def normalize_pos(pos: Sequence[float]) -> np.ndarray:
    return np.asarray(pos, dtype=np.float32) * 2.0 - 1.0


def circle_in_bounds(center: Sequence[float], radius: float) -> bool:
    x, y = center
    return radius <= x <= 1.0 - radius and radius <= y <= 1.0 - radius


def circle_rect_overlap(center: Sequence[float], radius: float, rect: Rect) -> bool:
    x, y = center
    closest_x = min(max(float(x), rect.x1), rect.x2)
    closest_y = min(max(float(y), rect.y1), rect.y2)
    return distance((x, y), (closest_x, closest_y)) < radius


def circle_any_rect_overlap(center: Sequence[float], radius: float, rects: Iterable[Rect]) -> bool:
    return any(circle_rect_overlap(center, radius, rect) for rect in rects)


def circle_overlap(a: Sequence[float], ra: float, b: Sequence[float], rb: float, margin: float = 1e-6) -> bool:
    return distance(a, b) < (ra + rb - margin)


def point_to_segment_distance(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    p = np.asarray(point, dtype=np.float32)
    start = np.asarray(a, dtype=np.float32)
    end = np.asarray(b, dtype=np.float32)
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-12:
        return distance(p, start)
    t = float(np.clip(np.dot(p - start, seg) / denom, 0.0, 1.0))
    projection = start + t * seg
    return distance(p, projection)


def orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c
    return (by - ay) * (cx - bx) - (bx - ax) * (cy - by)


def segments_intersect(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def segment_distance(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_to_segment_distance(a, c, d),
        point_to_segment_distance(b, c, d),
        point_to_segment_distance(c, a, b),
        point_to_segment_distance(d, a, b),
    )


def grip_points(center: Sequence[float], object_radius: float, agent_radius: float) -> dict[str, np.ndarray]:
    offset = object_radius + agent_radius + 0.014
    c = np.asarray(center, dtype=np.float32)
    return {
        "left": c + vec(-offset, 0.0),
        "right": c + vec(offset, 0.0),
        "top": c + vec(0.0, -offset),
        "bottom": c + vec(0.0, offset),
    }


def opposite_sides(side_a: str, side_b: str) -> bool:
    return {side_a, side_b} in ({"left", "right"}, {"top", "bottom"})


def nearest_rect_distance(center: Sequence[float], rects: Iterable[Rect]) -> float:
    rects = list(rects)
    if not rects:
        return 1.0
    x, y = center
    best = 1.0
    for rect in rects:
        closest_x = min(max(float(x), rect.x1), rect.x2)
        closest_y = min(max(float(y), rect.y1), rect.y2)
        best = min(best, distance((x, y), (closest_x, closest_y)))
    return best
