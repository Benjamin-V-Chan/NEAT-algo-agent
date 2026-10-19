from __future__ import annotations

import colorsys
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neat_racer.config import RenderingSection
from neat_racer.simulation.track import Track

# --- Palette (dark racing-telemetry theme) ------------------------------------
BG_TOP = (20, 25, 34)
BG_BOTTOM = (8, 10, 14)
ASPHALT = (44, 49, 58)
KERB = (214, 220, 230)
DEAD_CAR = (74, 80, 90)
LEAD = (255, 212, 92)
RAY = (132, 194, 238)
PANEL_BG = (12, 15, 20)
FG = (230, 236, 244)
DIM = (142, 152, 164)
GOOD = (120, 222, 150)


@dataclass(slots=True)
class RenderControls:
    quit_requested: bool = False
    paused: bool = False
    draw_sensors: bool = True
    draw_trails: bool = True
    ticks_per_frame: int = 3


def _car_color(genome_id: int) -> tuple[int, int, int]:
    hue = (genome_id * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.60, 0.99)
    return int(r * 255), int(g * 255), int(b * 255)


class LiveRenderer:
    def __init__(self, cfg: RenderingSection) -> None:
        self.cfg = cfg
        self._pygame = None
        self._gfxdraw = None
        self.screen = None
        self.font = None
        self.font_small = None
        self.font_big = None
        self.clock = None
        self.capture = False
        self.controls = RenderControls(
            draw_sensors=cfg.draw_sensors,
            draw_trails=cfg.draw_trails,
            ticks_per_frame=max(1, cfg.sim_ticks_per_frame),
        )
        self.world_scale = 1.0
        self.offset = np.zeros(2, dtype=float)
        self._bg = None  # cached static-track background surface
        self._ray_layer = None  # reused SRCALPHA overlay for sensor rays
        self._car_layer = None  # reused SRCALPHA overlay for dead cars
        self._trail_id: int | None = None
        self._trail: deque = deque(maxlen=70)

    # ---- lifecycle -------------------------------------------------------
    def initialize(self, track: Track, capture: bool = False) -> None:
        if capture:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Live rendering requires the optional 'viz' extra. Install it with:  pip install -e '.[viz]'"
            ) from exc

        pygame.init()
        self._pygame = pygame
        self._gfxdraw = gfxdraw
        self.capture = capture
        size = (self.cfg.width, self.cfg.height)
        if capture:
            self.screen = pygame.Surface(size)
        else:
            pygame.display.set_caption("NEAT Racer — population")
            self.screen = pygame.display.set_mode(size)
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.font_small = pygame.font.SysFont("consolas,menlo,monospace", 12)
        self.font_big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)
        self.clock = pygame.time.Clock()
        self._compute_projection(track)
        self._ray_layer = pygame.Surface(size, pygame.SRCALPHA)
        self._car_layer = pygame.Surface(size, pygame.SRCALPHA)
        self._bg = self._build_background(track)

    def shutdown(self) -> None:
        if self._pygame:
            self._pygame.quit()

    def _compute_projection(self, track: Track) -> None:
        points = np.vstack([track.outer_boundary, track.inner_boundary])
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        margin = 44.0
        viewport_w = self.cfg.width - 2 * margin
        viewport_h = self.cfg.height - 2 * margin
        self.world_scale = float(min(viewport_w / span[0], viewport_h / span[1]))
        drawn = span * self.world_scale
        pad = (np.array([self.cfg.width, self.cfg.height], dtype=float) - drawn) * 0.5
        self.offset = pad - min_xy * self.world_scale

    def to_screen(self, p: np.ndarray) -> tuple[int, int]:
        sp = p * self.world_scale + self.offset
        return int(sp[0]), int(sp[1])

    # ---- static background (built once) ----------------------------------
    def _gradient_surface(self):
        pygame = self._pygame
        w, h = self.cfg.width, self.cfg.height
        grad = pygame.Surface((w, h))
        for y in range(h):
            t = y / max(1, h - 1)
            col = tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3))
            pygame.draw.line(grad, col, (0, y), (w, y))
        return grad

    def _build_background(self, track: Track):
        pygame = self._pygame
        gfx = self._gfxdraw
        w, h = self.cfg.width, self.cfg.height

        bg = self._gradient_surface()
        outer = [self.to_screen(p) for p in track.outer_boundary]
        inner = [self.to_screen(p) for p in track.inner_boundary]

        # Asphalt corridor over the whole outer ring.
        gfx.filled_polygon(bg, outer, ASPHALT)
        gfx.aapolygon(bg, outer, ASPHALT)

        # Punch the infield back to the backdrop so only the track reads as track:
        # copy the pure gradient, keep it only inside the inner polygon via an alpha mask,
        # then alpha-blit it back over the asphalt.
        cut = pygame.Surface((w, h), pygame.SRCALPHA)
        cut.blit(bg, (0, 0))  # asphalt+gradient, fully opaque
        grad = self._gradient_surface()
        cut.blit(grad, (0, 0))  # restore pure gradient everywhere (opaque)
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        gfx.filled_polygon(mask, inner, (255, 255, 255, 255))
        cut.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)  # keep only infield
        bg.blit(cut, (0, 0))

        # Kerbs: crisp light edges on both boundaries.
        self._aa_closed_polyline(bg, outer, KERB, 3)
        self._aa_closed_polyline(bg, inner, KERB, 3)

        self._draw_start_finish(bg, track)
        return bg

    def _aa_closed_polyline(self, surface, pts, color, width) -> None:
        pygame = self._pygame
        pygame.draw.lines(surface, color, True, pts, width)
        self._gfxdraw.aapolygon(surface, pts, color)

    def _draw_start_finish(self, surface, track: Track) -> None:
        pygame = self._pygame
        a = np.asarray(track.start_finish[0], dtype=float)
        b = np.asarray(track.start_finish[1], dtype=float)
        pa, pb = np.array(self.to_screen(a)), np.array(self.to_screen(b))
        vec = pb - pa
        length = float(np.hypot(*vec))
        if length < 1.0:
            return
        unit = vec / length
        normal = np.array([-unit[1], unit[0]])
        n_squares = max(6, int(length / 7))
        sq = length / n_squares
        for row in range(2):
            for i in range(n_squares):
                if (i + row) % 2 == 0:
                    continue
                base = pa + unit * (i * sq) + normal * (row * sq - sq)
                quad = [
                    base, base + unit * sq, base + unit * sq + normal * sq, base + normal * sq,
                ]
                pygame.draw.polygon(surface, (238, 240, 244), [(int(p[0]), int(p[1])) for p in quad])

    # ---- input -----------------------------------------------------------
    def poll_events(self) -> RenderControls:
        pygame = self._pygame
        if not pygame or self.capture:
            return self.controls
        c = self.controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                c.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    c.quit_requested = True
                elif event.key == pygame.K_SPACE:
                    c.paused = not c.paused
                elif event.key == pygame.K_s:
                    c.draw_sensors = not c.draw_sensors
                elif event.key == pygame.K_t:
                    c.draw_trails = not c.draw_trails
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PLUS, pygame.K_EQUALS):
                    c.ticks_per_frame = min(30, c.ticks_per_frame + 1)
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    c.ticks_per_frame = max(1, c.ticks_per_frame - 1)
        return c

    def frame(self, track: Track, agents: list, overlay: dict[str, Any], tick: int) -> RenderControls:
        c = self.poll_events()
        if c.quit_requested:
            return c
        while c.paused and not c.quit_requested:
            self.draw_population(track, agents, overlay)
            c = self.poll_events()
        if c.quit_requested:
            return c
        if tick % max(1, c.ticks_per_frame) == 0:
            self.draw_population(track, agents, overlay)
        return c

    # ---- drawing ---------------------------------------------------------
    def draw_population(self, track: Track, agents: list, overlay: dict[str, Any]) -> None:
        if not self.screen or not self._pygame:
            return
        pygame = self._pygame
        self.screen.blit(self._bg, (0, 0))

        lead = None
        best = float("-inf")
        for a in agents:
            if a.alive and a.fitness_total > best:
                best = a.fitness_total
                lead = a

        if self.controls.draw_sensors:
            self._draw_rays(agents)
        if self.controls.draw_trails:
            self._draw_leader_trail(lead)

        # Dead cars first (underneath) on one reused layer, then alive cars, leader on top.
        self._draw_dead_cars(agents)
        for a in agents:
            if a.alive and a is not lead:
                self._draw_car(a, lead=False)
        if lead is not None:
            self._draw_car(lead, lead=True)

        self._draw_hud(overlay)

        if not self.capture:
            pygame.display.flip()
            self.clock.tick(self.cfg.target_fps)

    def save_frame(self, path: str | Path) -> None:
        if self.screen and self._pygame:
            self._pygame.image.save(self.screen, str(path))

    def _draw_rays(self, agents: list) -> None:
        pygame = self._pygame
        alive = [a for a in agents if a.alive and a.render_rays is not None]
        if not alive:
            return
        sensors = alive[0].cfg.sensors
        angles = sensors.angles_deg
        max_range = sensors.max_range
        layer = self._ray_layer
        layer.fill((0, 0, 0, 0))
        alpha = 10 if len(alive) > 20 else 22
        col = (*RAY, alpha)
        for a in alive:
            car = a.car
            origin = self.to_screen(car.pos)
            for dist_norm, angle_deg in zip(a.render_rays, angles, strict=False):
                theta = car.heading + math.radians(angle_deg)
                length = float(dist_norm) * max_range
                end = car.pos + np.array([math.cos(theta), math.sin(theta)]) * length
                pygame.draw.aaline(layer, col, origin, self.to_screen(end))
        self.screen.blit(layer, (0, 0))

    def _draw_leader_trail(self, lead) -> None:
        if lead is None:
            return
        if lead.genome_id != self._trail_id:
            self._trail_id = lead.genome_id
            self._trail.clear()
        self._trail.append(self.to_screen(lead.car.pos))
        if len(self._trail) < 2:
            return
        pygame = self._pygame
        layer = self._ray_layer  # borrow a scratch layer (rays already blitted)
        layer.fill((0, 0, 0, 0))
        pts = list(self._trail)
        n = len(pts)
        for i in range(1, n):
            a = int(140 * (i / n))
            pygame.draw.line(layer, (*LEAD, a), pts[i - 1], pts[i], 2)
        self.screen.blit(layer, (0, 0))

    def _draw_dead_cars(self, agents: list) -> None:
        dead = [a for a in agents if not a.alive]
        if not dead:
            return
        layer = self._car_layer
        layer.fill((0, 0, 0, 0))
        for a in dead:
            self._car_body(layer, a.car, (*DEAD_CAR, 110), cabin=None, lights=False)
        self.screen.blit(layer, (0, 0))

    def _draw_car(self, agent, lead: bool) -> None:
        car = agent.car
        color = _car_color(agent.genome_id)
        cabin = tuple(int(v * 0.42) for v in color)
        self._car_body(self.screen, car, color, cabin=cabin, lights=True)
        if lead:
            self._draw_leader_ring(car)

    def _car_body(self, surface, car, color, cabin, lights: bool) -> None:
        gfx = self._gfxdraw
        h = car.heading
        fwd = np.array([math.cos(h), math.sin(h)], dtype=float)
        side = np.array([-math.sin(h), math.cos(h)], dtype=float)
        c = car.pos
        half_len, half_w = 9.5, 5.0

        def pts(pairs):
            return [self.to_screen(c + fwd * fl + side * sw) for fl, sw in pairs]

        body = pts([(half_len, half_w), (half_len, -half_w), (-half_len, -half_w), (-half_len, half_w)])
        gfx.filled_polygon(surface, body, color)
        gfx.aapolygon(surface, body, color)
        if cabin is not None:
            cab = pts([(half_len * 0.25, half_w * 0.72), (half_len * 0.25, -half_w * 0.72),
                       (-half_len * 0.55, -half_w * 0.72), (-half_len * 0.55, half_w * 0.72)])
            gfx.filled_polygon(surface, cab, cabin)
            gfx.aapolygon(surface, cab, cabin)
        if lights:
            for s in (0.52, -0.52):
                hp = self.to_screen(c + fwd * half_len * 0.92 + side * half_w * s)
                gfx.filled_circle(surface, hp[0], hp[1], 2, (255, 246, 206))

    def _draw_leader_ring(self, car) -> None:
        gfx = self._gfxdraw
        cx, cy = self.to_screen(car.pos)
        r = int(9.5 * self.world_scale) + 8
        gfx.aacircle(self.screen, cx, cy, r, LEAD)
        gfx.aacircle(self.screen, cx, cy, r + 1, (*LEAD, 120))

    def _draw_hud(self, overlay: dict[str, Any]) -> None:
        pygame = self._pygame
        pad = 16
        card_w, card_h = 290, 196
        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card.fill((*PANEL_BG, 210))
        pygame.draw.rect(card, (255, 255, 255, 22), card.get_rect(), 1, border_radius=8)
        self.screen.blit(card, (pad, pad))

        x = pad + 16
        self._hud_y = pad + 14

        def txt(s, color=FG, font=None, step=21):
            self.screen.blit((font or self.font).render(str(s), True, color), (x, self._hud_y))
            self._hud_y += step

        alive = int(overlay.get("alive", 0))
        total = int(overlay.get("total", 0) or 1)
        best_lap = overlay.get("run_best_lap")
        best_lap_s = f"{best_lap:.2f}s" if best_lap else "—"

        txt("NEAT RACER", (255, 255, 255), self.font_big, step=30)
        txt(f"generation   {overlay.get('generation', '-')}")
        txt(f"alive        {alive} / {total}", GOOD if alive else DIM)
        # alive bar
        bar_x, bar_w, bar_h = x, card_w - 48, 6
        pygame.draw.rect(self.screen, (40, 46, 54), (bar_x, self._hud_y, bar_w, bar_h), border_radius=3)
        frac = max(0.0, min(1.0, alive / total))
        pygame.draw.rect(self.screen, GOOD, (bar_x, self._hud_y, int(bar_w * frac), bar_h), border_radius=3)
        self._hud_y += bar_h + 10
        txt(f"gen best     {overlay.get('gen_best_fitness', 0):.0f}")
        txt(f"run best     {overlay.get('run_best_fitness', 0):.0f}")
        txt(f"run best lap {best_lap_s}")
        txt(
            f"lead  v={overlay.get('lead_speed', 0):.0f}  lap={overlay.get('lead_lap', 0)}"
            f"  cp={overlay.get('lead_checkpoint', 0)}",
            DIM, self.font_small,
        )

        if not self.capture:
            speed = self.controls.ticks_per_frame
            fps = self.clock.get_fps() if self.clock else 0.0
            right = f"{speed}x  ·  {fps:4.0f} fps"
            surf = self.font_small.render(right, True, DIM)
            self.screen.blit(surf, (self.cfg.width - surf.get_width() - pad, pad + 4))
            hint = "SPACE pause   [ / ]  watch speed   S rays   T trails   ESC quit"
            hsurf = self.font_small.render(hint, True, DIM)
            self.screen.blit(hsurf, (pad, self.cfg.height - 24))
