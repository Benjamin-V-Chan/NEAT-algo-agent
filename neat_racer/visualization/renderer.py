"""Pygame-based renderer and live telemetry overlay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neat_racer.config import RenderingSection
from neat_racer.simulation.car import CarState
from neat_racer.simulation.track import Track
from neat_racer.telemetry.metrics_bus import MetricsBus


@dataclass(slots=True)
class RenderControls:
    quit_requested: bool = False
    paused: bool = False
    speed_multiplier: float = 1.0
    draw_sensors: bool = True


class LiveRenderer:
    def __init__(self, cfg: RenderingSection, metrics_bus: MetricsBus) -> None:
        self.cfg = cfg
        self.metrics_bus = metrics_bus
        self._pygame = None
        self.screen = None
        self.font = None
        self.clock = None
        self.controls = RenderControls(speed_multiplier=1.0, draw_sensors=cfg.draw_sensors)
        self.world_scale = 1.0
        self.offset = np.zeros(2, dtype=float)
        self.panel_x = cfg.width - cfg.hud_panel_width

    def initialize(self, track: Track) -> None:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Live rendering requires the optional 'viz' extra. "
                "Install it with:  pip install -e '.[viz]'"
            ) from exc

        pygame.init()
        pygame.display.set_caption("NEAT Racer")
        self.screen = pygame.display.set_mode((self.cfg.width, self.cfg.height))
        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()
        self._pygame = pygame

        self._compute_projection(track)

    def shutdown(self) -> None:
        if self._pygame:
            self._pygame.quit()

    def _compute_projection(self, track: Track) -> None:
        points = np.vstack([track.outer_boundary, track.inner_boundary])
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)

        viewport_w = self.panel_x - 40
        viewport_h = self.cfg.height - 40
        sx = viewport_w / span[0]
        sy = viewport_h / span[1]
        self.world_scale = float(min(sx, sy))
        self.offset = np.array([20.0, 20.0], dtype=float) - min_xy * self.world_scale

    def to_screen(self, p: np.ndarray) -> tuple[int, int]:
        sp = p * self.world_scale + self.offset
        return int(sp[0]), int(sp[1])

    def poll_events(self) -> RenderControls:
        pygame = self._pygame
        if not pygame:
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

    def draw_frame(
        self,
        track: Track,
        cars: list[CarState],
        selected_car_idx: int,
        sensors: np.ndarray | None,
        overlay: dict[str, Any],
    ) -> None:
        if not self.screen or not self._pygame:
            return

        pygame = self._pygame
        self.screen.fill((22, 25, 31))

        self._draw_track(track)
        self._draw_checkpoints(track)

        for idx, car in enumerate(cars):
            self._draw_car(car, is_selected=(idx == selected_car_idx))

        if sensors is not None and cars and self.controls.draw_sensors:
            self._draw_sensors(cars[selected_car_idx], sensors)

        self._draw_hud(overlay)
        pygame.display.flip()
        target_fps = int(max(10, min(480, 60 * self.controls.speed_multiplier)))
        self.clock.tick(target_fps)

    def _draw_track(self, track: Track) -> None:
        pygame = self._pygame
        outer = [self.to_screen(p) for p in track.outer_boundary]
        inner = [self.to_screen(p) for p in track.inner_boundary]
        pygame.draw.polygon(self.screen, (58, 62, 72), outer)
        pygame.draw.polygon(self.screen, (17, 19, 23), inner)
        pygame.draw.lines(self.screen, (210, 210, 210), True, outer, 3)
        pygame.draw.lines(self.screen, (210, 210, 210), True, inner, 3)

        sf = [self.to_screen(track.start_finish[0]), self.to_screen(track.start_finish[1])]
        pygame.draw.line(self.screen, (250, 90, 90), sf[0], sf[1], 4)

    def _draw_checkpoints(self, track: Track) -> None:
        if not self.cfg.draw_checkpoints:
            return
        pygame = self._pygame
        for gate in track.checkpoints:
            a, b = self.to_screen(gate[0]), self.to_screen(gate[1])
            pygame.draw.line(self.screen, (70, 170, 255), a, b, 1)

    def _draw_car(self, car: CarState, is_selected: bool) -> None:
        pygame = self._pygame
        color = (255, 195, 65) if is_selected else (80, 220, 130)

        heading_vec = np.array([math.cos(car.heading), math.sin(car.heading)], dtype=float)
        right_vec = np.array([-heading_vec[1], heading_vec[0]], dtype=float)

        front = car.pos + heading_vec * 10
        rear_left = car.pos - heading_vec * 7 + right_vec * 5
        rear_right = car.pos - heading_vec * 7 - right_vec * 5

        points = [self.to_screen(front), self.to_screen(rear_left), self.to_screen(rear_right)]
        pygame.draw.polygon(self.screen, color, points)

    def _draw_sensors(self, car: CarState, sensors: np.ndarray) -> None:
        pygame = self._pygame
        if sensors is None:
            return

        angles = self.metrics_bus.car_metrics.get("sensor_angles_deg", [])
        for d, angle_deg in zip(sensors, angles, strict=False):
            ray_theta = car.heading + math.radians(angle_deg)
            length = float(d) * self.metrics_bus.car_metrics.get("sensor_max_range", 100.0)
            end = car.pos + np.array([math.cos(ray_theta), math.sin(ray_theta)]) * length
            pygame.draw.line(self.screen, (255, 233, 120), self.to_screen(car.pos), self.to_screen(end), 1)

    def _draw_hud(self, overlay: dict[str, Any]) -> None:
        pygame = self._pygame
        panel_rect = pygame.Rect(self.panel_x, 0, self.cfg.hud_panel_width, self.cfg.height)
        pygame.draw.rect(self.screen, (10, 12, 15), panel_rect)

        snapshot = self.metrics_bus.snapshot()
        lines = [
            "NEAT Racer",
            f"Gen: {snapshot['generation'].get('generation', '-')}",
            f"Genome: {snapshot['car'].get('genome_id', '-')}",
            f"Alive cars: {snapshot['generation'].get('alive_count', '-')}",
            f"Best fitness (gen): {snapshot['generation'].get('best_fitness', 0):.2f}",
            f"Best fitness (run): {snapshot['run'].get('best_fitness', 0):.2f}",
            f"Farthest progress: {snapshot['generation'].get('farthest_progress', 0):.3f}",
            f"Best lap (gen): {snapshot['generation'].get('best_lap_time', float('nan'))}",
            f"Best lap (run): {snapshot['run'].get('best_lap_time', float('nan'))}",
            f"Mean fit: {snapshot['generation'].get('mean_fitness', 0):.2f}",
            f"Median fit: {snapshot['generation'].get('median_fitness', 0):.2f}",
            f"Crash count: {snapshot['generation'].get('crash_count', 0)}",
            f"Speed: {snapshot['car'].get('speed', 0):.2f}",
            (
                f"Steer/T/B: {snapshot['car'].get('steer', 0):.2f} / "
                f"{snapshot['car'].get('throttle', 0):.2f} / {snapshot['car'].get('brake', 0):.2f}"
            ),
            f"Checkpoint: {snapshot['car'].get('checkpoint', 0)}",
            f"Lap: {snapshot['car'].get('lap', 0)}",
            f"Sim speed: x{self.controls.speed_multiplier:.2f}",
            "Keys: SPACE pause, +/- speed, S sensors",
        ]

        y = 16
        for line in lines:
            surface = self.font.render(str(line), True, (230, 230, 230))
            self.screen.blit(surface, (self.panel_x + 12, y))
            y += 24

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
            xs = [cp[0][0], cp[1][0]]
            ys = [cp[0][1], cp[1][1]]
            ax.plot(xs, ys, color="#3498db", linewidth=0.8)
        sf = track.start_finish
        ax.plot([sf[0][0], sf[1][0]], [sf[0][1], sf[1][1]], color="#e74c3c", linewidth=2)
        ax.set_aspect("equal")
        ax.set_title(track.name)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
