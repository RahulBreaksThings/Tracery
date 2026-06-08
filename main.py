import os
import sys
import cv2
from tracery import TraceryEffect, HandDetector


WINDOW = "Tracery"
HELP_LINES = [
    "[g] force-activate (camera)  [t] auto/manual  (manual: L-CLICK color, R-CLICK clear)",
    "[m] connection mode  [k] marker style  [a] arrows  [l] labels",
    "[e] edges  [d] dashed lines  [+/-] marker size",
    "[s] save snapshot  [r] reset  [q] quit",
]


def _draw_help(frame):
    h = frame.shape[0]
    y = h - 14 * len(HELP_LINES) - 8
    for line in HELP_LINES:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_PLAIN, 0.95,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y += 14


def _save_snapshot(frame, prefix="tracery"):
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    while True:
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        if not os.path.exists(path):
            break
        i += 1
    cv2.imwrite(path, frame)
    print(f"Saved {path}")


class _MouseState:
    def __init__(self):
        self.last_frame = None
        self.effect: TraceryEffect = None

    def callback(self, event, x, y, flags, _):
        if self.last_frame is None or self.effect is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = self.last_frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                bgr = tuple(int(c) for c in self.last_frame[y, x])
                self.effect.add_target_from_bgr(bgr)
                print(f"Added target color BGR={bgr} (total: {len(self.effect.targets)})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.effect.clear_targets()
            print("Cleared all targets.")


def _handle_key(key, effect: TraceryEffect, frame) -> str:
    if key == ord("q"):
        return "quit"
    if key == ord("m"):
        effect.cycle_mode()
        print(f"Connection mode: {effect.connection_mode}")
    elif key == ord("k"):
        effect.cycle_marker()
        print(f"Marker style: {effect.marker_style}")
    elif key == ord("a"):
        effect.show_arrows = not effect.show_arrows
    elif key == ord("l"):
        effect.show_labels = not effect.show_labels
    elif key == ord("e"):
        effect.show_edges = not effect.show_edges
    elif key == ord("d"):
        effect.dashed_lines = not effect.dashed_lines
    elif key == ord("t"):
        effect.auto_mode = not effect.auto_mode
        print(f"Auto mode: {effect.auto_mode}")
    elif key == ord("r"):
        effect.clear_targets()
    elif key in (ord("+"), ord("=")):
        effect.marker_size = min(80, effect.marker_size + 2)
    elif key == ord("-"):
        effect.marker_size = max(6, effect.marker_size - 2)
    elif key == ord("s"):
        _save_snapshot(frame)
    return ""


def _setup_window(effect, mouse_state):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    mouse_state.effect = effect
    cv2.setMouseCallback(WINDOW, mouse_state.callback)


def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return
    effect = TraceryEffect()
    detector = HandDetector()
    mouse = _MouseState()
    _setup_window(effect, mouse)
    activated = False
    print("Show an open hand once to activate the effect.  Press [g] to activate manually.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        mouse.last_frame = frame

        detected = detector.open_hand_visible(frame)
        if not activated and detected:
            activated = True
            print("Effect activated.")

        if activated:
            out = effect.process(frame)
            detector.draw_landmarks(out)
        else:
            out = frame.copy()

        _draw_help(out)
        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("g"):
            activated = True
            print("Effect activated (manual).")
            continue
        if _handle_key(key, effect, out) == "quit":
            break
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def run_image():
    path = input("Path to image file: ").strip().strip('"').strip("'")
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    frame = cv2.imread(path)
    if frame is None:
        print("OpenCV could not read that image.")
        return
    max_w = 1280
    if frame.shape[1] > max_w:
        scale = max_w / frame.shape[1]
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    effect = TraceryEffect()
    mouse = _MouseState()
    mouse.last_frame = frame
    _setup_window(effect, mouse)
    print("Auto-tracking colorful regions. Press [t] to switch to click-to-pick mode.")
    while True:
        out = effect.process(frame)
        _draw_help(out)
        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(30) & 0xFF
        if _handle_key(key, effect, out) == "quit":
            break
    cv2.destroyAllWindows()


def run_video():
    path = input("Path to video file: ").strip().strip('"').strip("'")
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save = input("Save processed video? [y/N]: ").strip().lower() == "y"
    writer = None
    if save:
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{base}_tracery.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Writing to {out_path}")

    effect = TraceryEffect()
    mouse = _MouseState()
    _setup_window(effect, mouse)
    delay = max(1, int(1000 / fps))
    paused = False
    last_frame = None
    print("Auto-tracking colorful regions. [SPACE] pause/resume. [t] toggle auto.")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame = frame
        mouse.last_frame = last_frame
        out = effect.process(last_frame)
        if writer is not None and not paused:
            writer.write(out)
        _draw_help(out)
        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord(" "):
            paused = not paused
            continue
        if _handle_key(key, effect, out) == "quit":
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def menu():
    while True:
        print("\n" + "=" * 44)
        print("  TRACERY  —  OpenCV recreation")
        print("=" * 44)
        print("  1) Camera (live)")
        print("  2) Image")
        print("  3) Video")
        print("  4) Exit")
        choice = input("Choose an option [1-4]: ").strip()
        if choice == "1":
            run_camera()
        elif choice == "2":
            run_image()
        elif choice == "3":
            run_video()
        elif choice == "4":
            print("Bye.")
            return
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
