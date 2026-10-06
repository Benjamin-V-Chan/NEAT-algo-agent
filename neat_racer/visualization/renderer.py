"""Pygame live renderer: the whole population racing at once.

Draws every car in the generation as a little car sprite, with a faint raycast "web" per car,
dead cars fading underneath, and a translucent telemetry card. Also supports an offscreen capture
mode (no window) so a frame can be rendered and saved to PNG on a headless machine.
"""

from __future__ import annotations

import colorsys
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neat_racer.config import RenderingSection
from neat_racer.simulation.track import Track
from neat_racer.telemetry.metrics_bus import MetricsBus

# Palette (dark telemetry theme)
BG = (14, 18, 22)
ASPHALT = (41, 46, 54)
ASPHALT_EDGE = (120, 130, 142)
KERB = (210, 214, 220)
CHECKPOINT = (70, 150, 230)
START_FINISH = (235, 92, 92)
DEAD_CAR = (70, 76, 84)
LEAD_RING = (255, 214, 90)
RAY = (120, 210, 255)
HUD_BG = (10, 13, 16)
HUD_FG = (228, 234, 240)
HUD_DIM = (150, 160, 170)


@dataclass(slots=True)
class RenderControls:
    quit_requested: bool = False
    paused: bool = False
    speed_multiplier: float = 1.0
    draw_sensors: bool = True


def _car_color(genome_id: int) -> tuple[int, int, int]:
    """A stable, well-spread color per car (golden-ratio hue hopping)."""
    hue = (genome_id * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.98)
    return int(r * 255), int(g * 255), int(b * 255)


class LiveRenderer:
    def __init__(self, cfg: RenderingSection, metrics_bus: MetricsBus) -> None:
        self.cfg = cfg
        self.metrics_bus = metrics_bus
        self._pygame = None
        self.screen = None
        self.font = None
        self.font_small = None
        self.font_big = None
        self.clock = None
        self.capture = False
        self.controls = RenderControls(speed_multiplier=1.0, draw_sensors=cfg.draw_sensors)
        self.world_scale = 1.0
        self.offset = np.zeros(2, dtype=float)

    # ---- lifecycle -------------------------------------------------------
    def initialize(self, track: Track, capture: bool = False) -> None:
        if capture:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Live rendering requires the optional 'viz' extra. Install it with:  pip install -e '.[viz]'"
            ) from exc

        pygame.init()
        self._pygame = pygame
        self.capture = capture
        size = (self.cfg.width, self.cfg.height)
        if capture:
            self.screen = pygame.Surface(size)
        else:
            pygame.display.set_caption("NEAT Racer — population")
            self.screen = pygame.display.set_mode(size)
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.font_small = pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.font_big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)
        self.clock = pygame.time.Clock()
        self._compute_projection(track)

    def shutdown(self) -> None:
        if self._pygame:
            self._pygame.quit()

    def _compute_projection(self, track: Track) -> None:
        points = np.vstack([track.outer_boundary, track.inner_boundary])
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        margin = 28.0
        viewport_w = self.cfg.width - 2 * margin
        viewport_h = self.cfg.height - 2 * margin
        self.world_scale = float(min(viewport_w / span[0], viewport_h / span[1]))
        # Center the track in the window.
        drawn = span * self.world_scale
        pad = (np.array([self.cfg.width, self.cfg.height], dtype=float) - drawn) * 0.5
        self.offset = pad - min_xy * self.world_scale

    def to_screen(self, p: np.ndarray) -> tuple[int, int]:
        sp = p * self.world_scale + self.offset
        return int(sp[0]), int(sp[1])

    # ---- input -----------------------------------------------------------
    def poll_events(self) -> RenderControls:
        pygame = self._pygame
        if not pygame or self.capture:
            return self.controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.controls.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.controls.paused = not self.controls.paused
                elif event.key == pygame.K_s:
                    self.controls.draw_sensors = not self.controls.draw_sensors
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.controls.speed_multiplier = min(16.0, self.controls.speed_multiplier * 1.5)
                elif event.key == pygame.K_MINUS:
                    self.controls.speed_multiplier = max(0.25, self.controls.speed_multiplier / 1.5)
        return self.controls

    # ---- drawing ---------------------------------------------------------
    def draw_population(self, track: Track, agents: list, overlay: dict[str, Any]) -> None:
        if not self.screen or not self._pygame:
            return
        pygame = self._pygame
        self.screen.fill(BG)
        self._draw_track(track)

        if self.controls.draw_sensors:
            self._draw_rays(agents)

        # Dead cars first (underneath), then alive, then the leader on top.
        lead = None
        best = float("-inf")
        for a in agents:
            if a.alive and a.fitness_total > best:
                best = a.fitness_total
                lead = a
        for a in agents:
            if not a.alive:
                self._draw_car(a, dead=True, lead=False)
        for a in agents:
            if a.alive and a is not lead:
                self._draw_car(a, dead=False, lead=False)
        if lead is not None:
            self._draw_car(lead, dead=False, lead=True)

        self._draw_hud(overlay)

        if not self.capture:
            pygame.display.flip()
            target_fps = int(max(10, min(480, 60 * self.controls.speed_multiplier)))
            self.clock.tick(target_fps)

    def save_frame(self, path: str | Path) -> None:
        if self.screen and self._pygame:
            self._pygame.image.save(self.screen, str(path))

    def _draw_track(self, track: Track) -> None:
        # Just the track: filled corridor + a single clean wall edge + the start/finish line.
        pygame = self._pygame
        outer = [self.to_screen(p) for p in track.outer_boundary]
        inner = [self.to_screen(p) for p in track.inner_boundary]
        pygame.draw.polygon(self.screen, ASPHALT, outer)
        pygame.draw.polygon(self.screen, BG, inner)
        pygame.draw.lines(self.screen, KERB, True, outer, 2)
        pygame.draw.lines(self.screen, KERB, True, inner, 2)
        sf = [self.to_screen(track.start_finish[0]), self.to_screen(track.start_finish[1])]
        pygame.draw.line(self.screen, START_FINISH, sf[0], sf[1], 4)

    def _draw_rays(self, agents: list) -> None:
        pygame = self._pygame
        alive = [a for a in agents if a.alive and a.render_rays is not None]
        if not alive:
            return
        sensors = alive[0].cfg.sensors
        angles = sensors.angles_deg
        max_range = sensors.max_range
        overlay = pygame.Surface((self.cfg.width, self.cfg.height), pygame.SRCALPHA)
        alpha = 26 if len(alive) > 20 else 55
        for a in alive:
            car = a.car
            origin = self.to_screen(car.pos)
            for dist_norm, angle_deg in zip(a.render_rays, angles, strict=False):
                theta = car.heading + math.radians(angle_deg)
                length = float(dist_norm) * max_range
                end = car.pos + np.array([math.cos(theta), math.sin(theta)]) * length
                pygame.draw.line(overlay, (*RAY, alpha), origin, self.to_screen(end), 1)
        self.screen.blit(overlay, (0, 0))

    def _draw_car(self, agent, dead: bool, lead: bool) -> None:
        pygame = self._pygame
        car = agent.car
        h = car.heading
        fwd = np.array([math.cos(h), math.sin(h)], dtype=float)
        side = np.array([-math.sin(h), math.cos(h)], dtype=float)
        c = car.pos
        half_len, half_w = 9.5, 5.2  # world units

        def poly(points):
            return [self.to_screen(p) for p in points]

        body = poly([
            c + fwd * half_len + side * half_w,
            c + fwd * half_len - side * half_w,
            c - fwd * half_len - side * half_w,
            c - fwd * half_len + side * half_w,
        ])
        if dead:
            surf = pygame.Surface((self.cfg.width, self.cfg.height), pygame.SRCALPHA)
            pygame.draw.polygon(surf, (*DEAD_CAR, 90), body)
            self.screen.blit(surf, (0, 0))
            return

        color = _car_color(agent.genome_id)
        pygame.draw.polygon(self.screen, color, body)
        # darker cabin toward the rear, lighter windshield toward the front
        cabin = poly([
            c + fwd * half_len * 0.2 + side * half_w * 0.7,
            c + fwd * half_len * 0.2 - side * half_w * 0.7,
            c - fwd * half_len * 0.6 - side * half_w * 0.7,
            c - fwd * half_len * 0.6 + side * half_w * 0.7,
        ])
        dark = tuple(int(v * 0.45) for v in color)
        pygame.draw.polygon(self.screen, dark, cabin)
        # headlights
        for s in (0.55, -0.55):
            hp = self.to_screen(c + fwd * half_len * 0.95 + side * half_w * s)
            pygame.draw.circle(self.screen, (255, 245, 200), hp, 2)
        if lead:
            pygame.draw.polygon(self.screen, LEAD_RING, body, 2)
            ring = self.to_screen(c)
            pygame.draw.circle(self.screen, LEAD_RING, ring, int(half_len * self.world_scale) + 7, 2)

    def _draw_hud(self, overlay: dict[str, Any]) -> None:
        pygame = self._pygame
        pad = 14
        card_w, card_h = 268, 172
        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card.fill((*HUD_BG, 205))
        pygame.draw.rect(card, (255, 255, 255, 18), card.get_rect(), 1)
        self.screen.blit(card, (pad, pad))

        x = pad + 14
        self._hud_y = pad + 12

        def txt(s, color=HUD_FG, font=None, step=22):
            self.screen.blit((font or self.font).render(str(s), True, color), (x, self._hud_y))
            self._hud_y += step

        best_lap = overlay.get("run_best_lap")
        best_lap_s = f"{best_lap:.2f}s" if best_lap else "—"
        alive = overlay.get("alive", 0)
        total = overlay.get("total", 0)
        txt("NEAT RACER", (255, 255, 255), self.font_big, step=32)
        txt(f"generation   {overlay.get('generation', '-')}")
        txt(f"alive        {alive} / {total}", (120, 220, 150) if alive else HUD_DIM)
        txt(f"gen best     {overlay.get('gen_best_fitness', 0):.0f}")
        txt(f"run best     {overlay.get('run_best_fitness', 0):.0f}")
        txt(f"run best lap {best_lap_s}")
        txt(
            f"lead  v={overlay.get('lead_speed', 0):.0f}  lap={overlay.get('lead_lap', 0)}"
            f"  cp={overlay.get('lead_checkpoint', 0)}",
            HUD_DIM, self.font_small,
        )

        if not self.capture:
            hint = "SPACE pause   +/- speed   S rays"
            surf = self.font_small.render(hint, True, HUD_DIM)
            self.screen.blit(surf, (pad, self.cfg.height - 24))

    # ---- static snapshot (matplotlib, used by the trainer) ---------------
    def save_track_snapshot(self, track: Track, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
        outer = np.vstack([track.outer_boundary, track.outer_boundary[0]])
        inner = np.vstack([track.inner_boundary, track.inner_boundary[0]])
        ax.plot(outer[:, 0], outer[:, 1], color="black", linewidth=2)
        ax.plot(inner[:, 0], inner[:, 1], color="black", linewidth=2)
        for cp in track.checkpoints:
            ax.plot([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], color="#3498db", linewidth=0.8)
        sf = track.start_finish
        ax.plot([sf[0][0], sf[1][0]], [sf[0][1], sf[1][1]], color="#e74c3c", linewidth=2)
        ax.set_aspect("equal")
        ax.set_title(track.name)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
