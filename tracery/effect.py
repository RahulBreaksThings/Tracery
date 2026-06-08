import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import heapq
import math


HUD_PALETTE = [
    (255, 255, 255),
    (80, 255, 200),
    (255, 200, 80),
    (255, 120, 200),
    (120, 200, 255),
    (180, 255, 120),
    (255, 255, 120),
    (200, 120, 255),
]

LABEL_COLOR = (0, 255, 255)

CONNECTION_MODES = ("sequential", "star", "full", "mst")
MARKER_STYLES = ("box", "dot", "cross", "plus", "reticle")


@dataclass
class ColorTarget:
    """A single color the user picked. We track every blob matching it."""
    hsv: Tuple[int, int, int]
    tolerance: Tuple[int, int, int] = (10, 60, 60)
    overlay_color: Tuple[int, int, int] = (255, 255, 255)
    label: str = "TGT"

    def mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        h, s, v = self.hsv
        dh, ds, dv = self.tolerance
        lower = np.array([max(0, h - dh), max(0, s - ds), max(0, v - dv)], np.uint8)
        upper = np.array([min(179, h + dh), min(255, s + ds), min(255, v + dv)], np.uint8)
        m = cv2.inRange(hsv_frame, lower, upper)
        if h - dh < 0 or h + dh > 179:
            wrap_lo = (h - dh) % 180
            wrap_hi = (h + dh) % 180
            lo2 = np.array([min(wrap_lo, wrap_hi), lower[1], lower[2]], np.uint8)
            hi2 = np.array([max(wrap_lo, wrap_hi), upper[1], upper[2]], np.uint8)
            m = cv2.bitwise_or(m, cv2.inRange(hsv_frame, lo2, hi2))
        return m


@dataclass
class TrackedPoint:
    x: int
    y: int
    area: int
    target_index: int


class TraceryEffect:
    """
    Recreation of the AEScripts Tracery look:
      - Track every region matching one or more user-picked colors
      - Draw HUD-style markers (box, dot, cross, plus, reticle) at each centroid
      - Connect points using Sequential / Star / Full / MST(Prim's) modes
      - Optional directional arrows, labels, and corner brackets
    """

    def __init__(
        self,
        connection_mode: str = "mst",
        marker_style: str = "box",
        marker_size: int = 18,
        line_thickness: int = 1,
        show_arrows: bool = False,
        show_labels: bool = True,
        show_brackets: bool = True,
        show_edges: bool = False,
        dashed_lines: bool = True,
        line_color: Tuple[int, int, int] = (255, 140, 30),
        auto_mode: bool = True,
        auto_sat_threshold: int = 70,
        auto_val_min: int = 60,
        auto_min_area: int = 250,
        auto_max_points: int = 24,
        auto_color: Tuple[int, int, int] = (255, 255, 255),
        feature_quality: float = 0.04,
        feature_min_distance: int = 40,
        feature_block_size: int = 9,
        max_targets_per_color: int = 64,
        min_area: int = 25,
        dim_background: float = 0.45,
        edge_low: int = 60,
        edge_high: int = 160,
        edge_tint: Tuple[float, float, float] = (0.9, 0.55, 0.15),
        edge_strength: float = 0.55,
        label_color: Tuple[int, int, int] = LABEL_COLOR,
        dash_length: int = 8,
        dash_gap: int = 6,
    ):
        self.targets: List[ColorTarget] = []
        self.connection_mode = connection_mode
        self.marker_style = marker_style
        self.marker_size = marker_size
        self.line_thickness = line_thickness
        self.show_arrows = show_arrows
        self.show_labels = show_labels
        self.show_brackets = show_brackets
        self.show_edges = show_edges
        self.dashed_lines = dashed_lines
        self.line_color = line_color
        self.auto_mode = auto_mode
        self.auto_sat_threshold = auto_sat_threshold
        self.auto_val_min = auto_val_min
        self.auto_min_area = auto_min_area
        self.auto_max_points = auto_max_points
        self.auto_color = auto_color
        self.feature_quality = feature_quality
        self.feature_min_distance = feature_min_distance
        self.feature_block_size = feature_block_size
        self.max_targets_per_color = max_targets_per_color
        self.min_area = min_area
        self.dim_background = dim_background
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.edge_tint = edge_tint
        self.edge_strength = edge_strength
        self.label_color = label_color
        self.dash_length = dash_length
        self.dash_gap = dash_gap
        self._next_palette_idx = 0

    def add_target_from_bgr(self, bgr: Tuple[int, int, int], tolerance=(15, 90, 90)):
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        color = HUD_PALETTE[self._next_palette_idx % len(HUD_PALETTE)]
        self._next_palette_idx += 1
        self.targets.append(
            ColorTarget(
                hsv=tuple(int(x) for x in hsv),
                tolerance=tolerance,
                overlay_color=color,
                label=f"T{len(self.targets):02d}",
            )
        )

    def clear_targets(self):
        self.targets.clear()
        self._next_palette_idx = 0

    def cycle_mode(self):
        i = CONNECTION_MODES.index(self.connection_mode)
        self.connection_mode = CONNECTION_MODES[(i + 1) % len(CONNECTION_MODES)]

    def cycle_marker(self):
        i = MARKER_STYLES.index(self.marker_style)
        self.marker_style = MARKER_STYLES[(i + 1) % len(MARKER_STYLES)]

    def _detect_auto(self, frame: np.ndarray) -> List[TrackedPoint]:
        """
        Shi-Tomasi corner detection: locks onto sharp features in the image
        (fingertips, knuckles, object corners, edge intersections) — i.e. places
        where image gradient is strong in two directions, not just one.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.auto_max_points,
            qualityLevel=self.feature_quality,
            minDistance=self.feature_min_distance,
            blockSize=self.feature_block_size,
            useHarrisDetector=False,
        )
        points: List[TrackedPoint] = []
        if corners is None:
            return points
        for c in corners:
            x, y = c.ravel()
            points.append(TrackedPoint(int(x), int(y), 1, -1))
        return points

    def _detect(self, frame: np.ndarray) -> List[TrackedPoint]:
        if not self.targets:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        all_points: List[TrackedPoint] = []
        for ti, target in enumerate(self.targets):
            m = target.mask(hsv)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            picks: List[TrackedPoint] = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                picks.append(TrackedPoint(cx, cy, int(area), ti))
            picks.sort(key=lambda p: -p.area)
            all_points.extend(picks[: self.max_targets_per_color])
        return all_points

    def _draw_marker(self, canvas, p: TrackedPoint, color):
        s = self.marker_size
        x, y = p.x, p.y
        t = self.line_thickness
        style = self.marker_style
        if style == "box":
            cv2.rectangle(canvas, (x - s, y - s), (x + s, y + s), color, t, cv2.LINE_AA)
            if self.show_brackets:
                b = max(4, s // 3)
                for dx, dy in [(-s, -s), (s, -s), (-s, s), (s, s)]:
                    cv2.line(canvas, (x + dx, y + dy),
                             (x + dx + (b if dx < 0 else -b), y + dy), color, t + 1, cv2.LINE_AA)
                    cv2.line(canvas, (x + dx, y + dy),
                             (x + dx, y + dy + (b if dy < 0 else -b)), color, t + 1, cv2.LINE_AA)
        elif style == "dot":
            cv2.circle(canvas, (x, y), max(3, s // 3), color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), s, color, t, cv2.LINE_AA)
        elif style == "cross":
            cv2.line(canvas, (x - s, y - s), (x + s, y + s), color, t, cv2.LINE_AA)
            cv2.line(canvas, (x - s, y + s), (x + s, y - s), color, t, cv2.LINE_AA)
        elif style == "plus":
            cv2.line(canvas, (x - s, y), (x + s, y), color, t, cv2.LINE_AA)
            cv2.line(canvas, (x, y - s), (x, y + s), color, t, cv2.LINE_AA)
        elif style == "reticle":
            cv2.circle(canvas, (x, y), s, color, t, cv2.LINE_AA)
            gap = s // 3
            cv2.line(canvas, (x - s - 6, y), (x - gap, y), color, t, cv2.LINE_AA)
            cv2.line(canvas, (x + gap, y), (x + s + 6, y), color, t, cv2.LINE_AA)
            cv2.line(canvas, (x, y - s - 6), (x, y - gap), color, t, cv2.LINE_AA)
            cv2.line(canvas, (x, y + gap), (x, y + s + 6), color, t, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)

    def _draw_label(self, canvas, p: TrackedPoint):
        if p.target_index < 0:
            label = "AUTO"
        else:
            label = self.targets[p.target_index].label
        text = f"{label}-{p.x:04d}.{p.y:04d}"
        off = self.marker_size + 6
        cv2.putText(canvas, text, (p.x + off, p.y - off // 2),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, self.label_color, 1, cv2.LINE_AA)

    def _point_color(self, p: TrackedPoint) -> Tuple[int, int, int]:
        if p.target_index < 0:
            return self.auto_color
        return self.targets[p.target_index].overlay_color

    def _draw_dashed(self, canvas, start, end, color):
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1:
            return
        ux, uy = dx / dist, dy / dist
        seg = self.dash_length + self.dash_gap
        n = int(dist // seg)
        for i in range(n + 1):
            s = i * seg
            e = min(s + self.dash_length, dist)
            if e <= s:
                continue
            p1 = (int(x1 + ux * s), int(y1 + uy * s))
            p2 = (int(x1 + ux * e), int(y1 + uy * e))
            cv2.line(canvas, p1, p2, color, self.line_thickness, cv2.LINE_AA)

    def _draw_line(self, canvas, a: TrackedPoint, b: TrackedPoint, color):
        pa = (a.x, a.y)
        pb = (b.x, b.y)
        gap = self.marker_size + 2
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        dist = math.hypot(dx, dy)
        if dist < gap * 2 + 4:
            return
        ux, uy = dx / dist, dy / dist
        start = (int(pa[0] + ux * gap), int(pa[1] + uy * gap))
        end = (int(pb[0] - ux * gap), int(pb[1] - uy * gap))
        if self.show_arrows:
            if self.dashed_lines:
                tip = (int(end[0] - ux * 10), int(end[1] - uy * 10))
                self._draw_dashed(canvas, start, tip, color)
                cv2.arrowedLine(canvas, tip, end, color, self.line_thickness,
                                cv2.LINE_AA, tipLength=0.6)
            else:
                cv2.arrowedLine(canvas, start, end, color, self.line_thickness,
                                cv2.LINE_AA, tipLength=0.04)
        else:
            if self.dashed_lines:
                self._draw_dashed(canvas, start, end, color)
            else:
                cv2.line(canvas, start, end, color, self.line_thickness, cv2.LINE_AA)

    def _connections(self, points: List[TrackedPoint]) -> List[Tuple[int, int]]:
        n = len(points)
        if n < 2:
            return []
        mode = self.connection_mode
        if mode == "sequential":
            return [(i, i + 1) for i in range(n - 1)]
        if mode == "star":
            return [(0, i) for i in range(1, n)]
        if mode == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        if mode == "mst":
            return self._prim_mst(points)
        return []

    @staticmethod
    def _prim_mst(points: List[TrackedPoint]) -> List[Tuple[int, int]]:
        n = len(points)
        in_tree = [False] * n
        in_tree[0] = True
        edges: List[Tuple[int, int]] = []
        heap: List[Tuple[float, int, int]] = []
        for j in range(1, n):
            d = math.hypot(points[0].x - points[j].x, points[0].y - points[j].y)
            heapq.heappush(heap, (d, 0, j))
        while heap and len(edges) < n - 1:
            d, i, j = heapq.heappop(heap)
            if in_tree[j]:
                continue
            in_tree[j] = True
            edges.append((i, j))
            for k in range(n):
                if not in_tree[k]:
                    dk = math.hypot(points[j].x - points[k].x, points[j].y - points[k].y)
                    heapq.heappush(heap, (dk, j, k))
        return edges

    def _draw_hud(self, canvas, n_points: int):
        h, w = canvas.shape[:2]
        c = (200, 230, 230)
        cv2.rectangle(canvas, (8, 8), (w - 8, h - 8), c, 1, cv2.LINE_AA)
        b = 14
        for cx, cy in [(8, 8), (w - 8, 8), (8, h - 8), (w - 8, h - 8)]:
            xs = b if cx == 8 else -b
            ys = b if cy == 8 else -b
            cv2.line(canvas, (cx, cy), (cx + xs, cy), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + ys), (255, 255, 255), 2, cv2.LINE_AA)
        src = "AUTO" if self.auto_mode else f"TARGETS:{len(self.targets)}"
        status = (f"TRACERY  MODE:{self.connection_mode.upper()}  "
                  f"MARKER:{self.marker_style.upper()}  "
                  f"{src}  POINTS:{n_points}")
        cv2.putText(canvas, status, (18, 26),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, c, 1, cv2.LINE_AA)

    def _edge_layer(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 1.0)
        edges = cv2.Canny(gray, self.edge_low, self.edge_high)
        edges_f = edges.astype(np.float32) / 255.0
        b, g, r = self.edge_tint
        layer = np.zeros_like(frame, dtype=np.float32)
        layer[..., 0] = edges_f * 255.0 * b
        layer[..., 1] = edges_f * 255.0 * g
        layer[..., 2] = edges_f * 255.0 * r
        return (layer * self.edge_strength).astype(np.uint8)

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.auto_mode:
            points = self._detect_auto(frame)
        else:
            points = self._detect(frame)

        if self.dim_background < 1.0:
            out = (frame.astype(np.float32) * self.dim_background).astype(np.uint8)
        else:
            out = frame.copy()

        if self.show_edges:
            out = cv2.add(out, self._edge_layer(frame))

        if len(points) >= 2:
            for a_idx, b_idx in self._connections(points):
                a, b = points[a_idx], points[b_idx]
                self._draw_line(out, a, b, self.line_color)

        for p in points:
            self._draw_marker(out, p, self._point_color(p))
            if self.show_labels:
                self._draw_label(out, p)

        self._draw_hud(out, len(points))
        if not self.auto_mode and not self.targets:
            self._draw_prompt(out)
        return out

    def _draw_prompt(self, canvas):
        h, w = canvas.shape[:2]
        msg1 = "LEFT-CLICK a colored object to start tracking it"
        msg2 = "(use a bright, distinct color for best results)"
        (tw1, th1), _ = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        (tw2, th2), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_PLAIN, 1.1, 1)
        cx = w // 2
        cy = h // 2
        pad = 16
        box_w = max(tw1, tw2) + pad * 2
        box_h = th1 + th2 + pad * 2 + 10
        x1 = cx - box_w // 2
        y1 = cy - box_h // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x1 + box_w, y1 + box_h),
                      self.label_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, msg1, (x1 + pad, y1 + pad + th1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.label_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, msg2, (x1 + pad, y1 + pad + th1 + 10 + th2),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (220, 220, 220), 1, cv2.LINE_AA)
