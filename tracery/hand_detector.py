import math
import os
import time
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "hand_landmarker.task")

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _ensure_model():
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000:
        return
    print(f"Downloading hand-landmark model -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class HandDetector:
    """
    Thin wrapper around MediaPipe's HandLandmarker (Tasks API).
    Exposes one signal: open_hand_visible(frame) -> bool.
    """

    def __init__(self, num_hands: int = 2, detection_confidence: float = 0.3,
                 spread_threshold: float = 1.35):
        _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self.last_result = None
        self.spread_threshold = spread_threshold

    @staticmethod
    def _non_thumb_fingers_extended(landmarks) -> int:
        count = 0
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            if landmarks[tip].y < landmarks[pip].y:
                count += 1
        return count

    @staticmethod
    def _spread_ratio(landmarks) -> float:
        """
        Ratio of (sum of adjacent fingertip distances) to (sum of adjacent
        knuckle distances). Fingers held together => fingertips track parallel
        to knuckles, ratio ~ 1.0. Fingers fanned out => fingertips diverge,
        ratio rises (~1.5-2.0 for a fully splayed hand).
        """
        tips = [landmarks[i] for i in (8, 12, 16, 20)]
        mcps = [landmarks[i] for i in (5, 9, 13, 17)]
        tip_span = sum(
            math.hypot(tips[i + 1].x - tips[i].x, tips[i + 1].y - tips[i].y)
            for i in range(3)
        )
        mcp_span = sum(
            math.hypot(mcps[i + 1].x - mcps[i].x, mcps[i + 1].y - mcps[i].y)
            for i in range(3)
        )
        if mcp_span < 1e-6:
            return 0.0
        return tip_span / mcp_span

    def open_hand_visible(self, frame_bgr: np.ndarray) -> bool:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        self.last_result = result
        if not result.hand_landmarks:
            return False
        for lms in result.hand_landmarks:
            if self._non_thumb_fingers_extended(lms) < 4:
                continue
            if self._spread_ratio(lms) >= self.spread_threshold:
                return True
        return False

    def draw_landmarks(self, frame: np.ndarray, color=(80, 220, 255)):
        if self.last_result is None or not self.last_result.hand_landmarks:
            return
        h, w = frame.shape[:2]
        for lms in self.last_result.hand_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in lms]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)

    def close(self):
        self._landmarker.close()
