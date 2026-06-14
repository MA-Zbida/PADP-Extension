"""Pygame renderer for the continuous warehouse carry environment."""
from __future__ import annotations

from typing import Sequence

import pygame

from environment.geometry import Rect, grip_points


class WarehouseRenderer:
    def __init__(self, size: int = 760, fps: int = 20) -> None:
        self.size = size
        self.fps = fps
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.small_font: pygame.font.Font | None = None

    def _px(self, pos: Sequence[float]) -> tuple[int, int]:
        return int(pos[0] * self.size), int(pos[1] * self.size)

    def _rect_px(self, rect: Rect) -> pygame.Rect:
        return pygame.Rect(
            int(rect.x1 * self.size),
            int(rect.y1 * self.size),
            int(rect.width * self.size),
            int(rect.height * self.size),
        )

    def render(self, env, debug: bool = False) -> None:
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.size, self.size))
            pygame.display.set_caption("Continuous Collaborative Warehouse")
            self.font = pygame.font.SysFont("Arial", 22, bold=True)
            self.small_font = pygame.font.SysFont("Arial", 15, bold=True)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((242, 243, 240))
        self._draw_zones(env)
        self._draw_racks(env)
        self._draw_goals(env)
        self._draw_carrier_links(env)
        self._draw_objects(env, debug=debug)
        self._draw_agents(env)
        self._draw_status(env)

        pygame.display.flip()
        pygame.event.pump()
        self.clock.tick(self.fps)

    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
            self.small_font = None

    def _draw_zones(self, env) -> None:
        assert self.screen is not None
        pygame.draw.rect(self.screen, (218, 234, 255), self._rect_px(env.layout.depot), border_radius=10)
        pygame.draw.rect(self.screen, (219, 243, 225), self._rect_px(env.layout.shipping), border_radius=10)
        for point in env.layout.pick_faces:
            pygame.draw.circle(self.screen, (255, 246, 206), self._px(point), int(0.018 * self.size))

        self._label_rect(env.layout.depot, "DEPOT", (70, 92, 120))
        self._label_rect(env.layout.shipping, "SHIP", (67, 110, 80))

    def _draw_racks(self, env) -> None:
        assert self.screen is not None
        for idx, rack in enumerate(env.racks):
            rect = self._rect_px(rack)
            pygame.draw.rect(self.screen, (94, 97, 96), rect, border_radius=5)
            pygame.draw.rect(self.screen, (62, 64, 64), rect, width=2, border_radius=5)
            self._text(f"R{idx + 1}", rack.center, (245, 245, 245), small=True)

    def _draw_goals(self, env) -> None:
        assert self.screen is not None
        for idx, goal in enumerate(env.goal_positions):
            color = (92, 176, 108) if env.goal_used[idx] else (40, 152, 82)
            center = self._px(goal)
            radius = int(env.goal_radius * self.size)
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, (22, 95, 56), center, radius, width=3)
            self._text(f"G{idx + 1}", goal, (255, 255, 255))

    def _draw_carrier_links(self, env) -> None:
        assert self.screen is not None
        for obj_idx, holders in enumerate(env.object_holders):
            if len(holders) != 2 or env.delivered[obj_idx]:
                continue
            a, b = holders
            start = self._px(env.agent_positions[a])
            end = self._px(env.agent_positions[b])
            pygame.draw.line(self.screen, (38, 105, 180), start, end, width=8)
            pygame.draw.line(self.screen, (191, 222, 255), start, end, width=3)

    def _draw_objects(self, env, debug: bool = False) -> None:
        assert self.screen is not None
        for idx, obj in enumerate(env.object_positions):
            held = len(env.object_holders[idx]) == 2
            delivered = env.delivered[idx]
            center = self._px(obj)
            radius = int(env.object_radius * self.size)
            color = (178, 183, 188) if delivered else ((246, 181, 68) if held else (246, 207, 54))
            rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=7)
            pygame.draw.rect(self.screen, (94, 80, 40), rect, width=2, border_radius=7)
            self._text(f"O{idx + 1}", obj, (25, 25, 25))

            if debug and not held and not delivered:
                for side, point in grip_points(obj, env.object_radius, env.agent_radius).items():
                    pygame.draw.circle(self.screen, (55, 130, 230), self._px(point), int(env.pickup_radius * self.size), width=1)
                    self._text(side[0].upper(), point, (55, 85, 150), small=True)

    def _draw_agents(self, env) -> None:
        assert self.screen is not None
        palette = [
            (54, 117, 181), (45, 145, 100), (142, 82, 210), (232, 96, 73),
            (226, 142, 48), (41, 156, 169), (156, 112, 60), (200, 83, 142),
            (90, 120, 210), (110, 160, 70),
        ]
        for idx, pos in enumerate(env.agent_positions):
            color = palette[idx % len(palette)]
            center = self._px(pos)
            radius = int(env.agent_radius * self.size)
            holding = env.agent_holding[idx] is not None
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, (255, 255, 255) if holding else (30, 30, 30), center, radius, width=3)
            self._text(str(idx + 1), pos, (255, 255, 255))

    def _draw_status(self, env) -> None:
        assert self.screen is not None
        collisions = env.last_events.agent_collisions + env.last_events.rack_collisions + env.last_events.bound_collisions
        if collisions > 0:
            pygame.draw.rect(self.screen, (220, 55, 55), pygame.Rect(4, 4, self.size - 8, self.size - 8), width=5)

    def _label_rect(self, rect: Rect, text: str, color: tuple[int, int, int]) -> None:
        self._text(text, (rect.x1 + 0.04, rect.y1 + 0.03), color, small=True)

    def _text(self, text: str, pos: Sequence[float], color: tuple[int, int, int], small: bool = False) -> None:
        assert self.screen is not None
        font = self.small_font if small else self.font
        assert font is not None
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=self._px(pos))
        self.screen.blit(surface, rect)
