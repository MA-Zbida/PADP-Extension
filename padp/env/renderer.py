"""Pygame renderer for the continuous warehouse (imported only when rendering)."""
from __future__ import annotations

from typing import Sequence

import pygame

from padp.env.geometry import Rect

BACKGROUND = (242, 243, 240)
DEPOT = (218, 234, 255)
SHIPPING = (219, 243, 225)
PICK_FACE = (255, 246, 206)
RACK, RACK_EDGE = (94, 97, 96), (62, 64, 64)
GOAL, GOAL_USED, GOAL_EDGE = (40, 152, 82), (92, 176, 108), (22, 95, 56)
OBJECT_FREE, OBJECT_LIFTED, OBJECT_DONE, OBJECT_EDGE = (246, 207, 54), (246, 181, 68), (178, 183, 188), (94, 80, 40)
LINK_ATTACHED, LINK_LIFTED = (150, 150, 150), (38, 105, 180)
BAR_BG, BAR_FILL, BAR_FULL = (210, 210, 205), (226, 142, 48), (45, 145, 100)
COLLISION = (220, 55, 55)
AGENT_COLORS = [
    (54, 117, 181), (45, 145, 100), (142, 82, 210), (232, 96, 73),
    (226, 142, 48), (41, 156, 169), (156, 112, 60), (200, 83, 142),
]


class WarehouseRenderer:
    def __init__(self, size: int = 760, fps: int = 20) -> None:
        self.size = size
        self.fps = fps
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("PADP warehouse")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 13, bold=True)

    def render(self, env, debug: bool = False) -> None:
        self.screen.fill(BACKGROUND)
        self._draw_zones(env)
        self._draw_racks(env)
        self._draw_goals(env)
        self._draw_links(env)
        self._draw_objects(env, debug)
        self._draw_agents(env)
        if env.last_events.collisions > 0:
            pygame.draw.rect(self.screen, COLLISION, pygame.Rect(4, 4, self.size - 8, self.size - 8), width=5)
        self._text(f"t={env.t}", (0.5, 0.025), (90, 90, 90), small=True)
        pygame.display.flip()
        pygame.event.pump()
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    # ------------------------------------------------------------------ helpers

    def _px(self, pos: Sequence[float]) -> tuple:
        return int(pos[0] * self.size), int(pos[1] * self.size)

    def _rect_px(self, rect: Rect) -> pygame.Rect:
        return pygame.Rect(int(rect.x1 * self.size), int(rect.y1 * self.size),
                           int(rect.width * self.size), int(rect.height * self.size))

    def _text(self, text: str, pos: Sequence[float], color: tuple, small: bool = False) -> None:
        surface = (self.small_font if small else self.font).render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=self._px(pos)))

    # ------------------------------------------------------------------ layers

    def _draw_zones(self, env) -> None:
        pygame.draw.rect(self.screen, DEPOT, self._rect_px(env.layout.depot), border_radius=10)
        pygame.draw.rect(self.screen, SHIPPING, self._rect_px(env.layout.shipping), border_radius=10)
        for point in env.layout.pick_faces:
            pygame.draw.circle(self.screen, PICK_FACE, self._px(point), int(0.018 * self.size))
        depot, ship = env.layout.depot, env.layout.shipping
        self._text("DEPOT", (depot.x1 + 0.05, depot.y1 + 0.03), (70, 92, 120), small=True)
        self._text("SHIP", (ship.x1 + 0.04, ship.y1 + 0.03), (67, 110, 80), small=True)

    def _draw_racks(self, env) -> None:
        for idx, rack in enumerate(env.racks):
            rect = self._rect_px(rack)
            pygame.draw.rect(self.screen, RACK, rect, border_radius=5)
            pygame.draw.rect(self.screen, RACK_EDGE, rect, width=2, border_radius=5)
            self._text(f"R{idx + 1}", rack.center, (245, 245, 245), small=True)

    def _draw_goals(self, env) -> None:
        radius = int(env.goal_radius * self.size)
        for idx, goal in enumerate(env.goal_positions):
            center = self._px(goal)
            pygame.draw.circle(self.screen, GOAL_USED if env.goal_used[idx] else GOAL, center, radius)
            pygame.draw.circle(self.screen, GOAL_EDGE, center, radius, width=3)
            self._text(f"G{idx + 1}", goal, (255, 255, 255))

    def _draw_links(self, env) -> None:
        for o, holders in enumerate(env.holders):
            if env.delivered[o]:
                continue
            color = LINK_LIFTED if env.lifted[o] else LINK_ATTACHED
            center = self._px(env.object_positions[o])
            for h in holders:
                pygame.draw.line(self.screen, color, self._px(env.agent_positions[h]), center, width=6)

    def _draw_objects(self, env, debug: bool) -> None:
        radius = int(env.object_radius * self.size)
        for o, pos in enumerate(env.object_positions):
            center = self._px(pos)
            color = OBJECT_DONE if env.delivered[o] else (OBJECT_LIFTED if env.lifted[o] else OBJECT_FREE)
            rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=7)
            pygame.draw.rect(self.screen, OBJECT_EDGE, rect, width=2, border_radius=7)
            self._text(f"{env.weights[o]:.0f}", pos, (25, 25, 25), small=True)
            if env.delivered[o]:
                continue
            # Attached-capacity bar along the object's bottom edge: orange while waiting, green when liftable.
            covered = min(1.0, env._attached_capacity(o) / env.weights[o])
            bar = pygame.Rect(center[0] - radius + 4, center[1] + radius - 9, radius * 2 - 8, 5)
            pygame.draw.rect(self.screen, BAR_BG, bar)
            fill = bar.copy()
            fill.width = int(bar.width * covered)
            pygame.draw.rect(self.screen, BAR_FULL if covered >= 1.0 else BAR_FILL, fill)
            if debug and not env.lifted[o]:
                for side, slot in env._free_slots(o).items():
                    pygame.draw.circle(self.screen, (55, 130, 230), self._px(slot), int(env.pickup_radius * self.size), width=1)
                    self._text(side[0].upper(), slot, (55, 85, 150), small=True)

    def _draw_agents(self, env) -> None:
        radius = int(env.agent_radius * self.size)
        max_cap = max(env.cfg.max_capacity, 1.0)
        for i, pos in enumerate(env.agent_positions):
            center = self._px(pos)
            pygame.draw.circle(self.screen, AGENT_COLORS[i % len(AGENT_COLORS)], center, radius)
            # Ring thickness encodes capacity.
            ring = 1 + int(4 * env.capacities[i] / max_cap)
            holding = env.agent_holding[i] is not None
            pygame.draw.circle(self.screen, (255, 255, 255) if holding else (30, 30, 30), center, radius, width=ring)
            self._text(str(i + 1), pos, (255, 255, 255), small=True)
            self._text(f"{env.capacities[i]:.0f}", (pos[0], pos[1] + env.agent_radius + 0.015), (60, 60, 60), small=True)
