"""Phase 0 — Camera Capture Baseline.

Opens a camera, shows a live mirrored preview, and measures its own
timing. No ML, no MediaPipe, no gesture logic. Establishes the latency
floor for the real-time hand-gesture shortcut system.

Usage:
    uv run capture.py [--camera INDEX] [--width W] [--height H] [--fps FPS]
"""

import argparse
import collections
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(buf, p):
    """Return the p-th percentile from a deque of floats."""
    if not buf:
        return 0.0
    arr = np.array(buf)
    return float(np.percentile(arr, p))


def ms(seconds):
    """Seconds → milliseconds, rounded to one decimal place."""
    return round(seconds * 1000, 1)


def print_capture_props(cap, requested):
    """Print requested vs actual camera properties side-by-side."""
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_buf = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))

    print("┌─────────────────────────────────────────────┐")
    print("│          Camera Properties                  │")
    print("├──────────────┬──────────┬────────────────────┤")
    print(f"│ {'Property':<12} │ {'Requested':>8} │ {'Actual':<18} │")
    print("├──────────────┼──────────┼────────────────────┤")
    print(f"│ {'Width':<12} │ {requested['width']:>8} │ {actual_w:<18} │")
    print(f"│ {'Height':<12} │ {requested['height']:>8} │ {actual_h:<18} │")
    print(f"│ {'FPS':<12} │ {requested['fps']:>8} │ {actual_fps:<18.1f} │")
    print(f"│ {'Buffer':<12} │ {1:>8} │ {actual_buf:<18} │")
    print("└──────────────┴──────────┴────────────────────┘")

    if actual_w != requested["width"] or actual_h != requested["height"]:
        print(
            f"⚠  Resolution mismatch: requested {requested['width']}×{requested['height']}, "
            f"got {actual_w}×{actual_h}"
        )

    return actual_w, actual_h, actual_fps


def draw_overlay(frame, intervals, read_times, actual_w, actual_h):
    """Draw live timing stats on the top-left of the frame."""
    p50_interval = ms(percentile(intervals, 50))
    p95_interval = ms(percentile(intervals, 95))
    p50_read = ms(percentile(read_times, 50))
    equiv_fps = round(1000.0 / p50_interval, 1) if p50_interval > 0 else 0.0

    lines = [
        f"interval  p50 {p50_interval:>6.1f} ms   p95 {p95_interval:>6.1f} ms",
        f"read      p50 {p50_read:>6.1f} ms",
        f"FPS (p50)     {equiv_fps:>6.1f}",
        f"resolution    {actual_w}x{actual_h}",
    ]

    y0 = 30
    for i, line in enumerate(lines):
        y = y0 + i * 28
        # Shadow for readability over any background
        cv2.putText(
            frame, line, (11, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------

SCAN_RANGE = 5  # indices 0-4

def open_camera(preferred_index):
    """Try to open a camera, with AVFoundation backend and auto-scan fallback.

    Returns an opened VideoCapture or None.
    """
    backend = cv2.CAP_AVFOUNDATION

    # Try the preferred index first
    cap = cv2.VideoCapture(preferred_index, backend)
    if cap.isOpened():
        print(f"✓ Opened camera index {preferred_index} (AVFoundation)")
        return cap
    cap.release()

    # Auto-scan other indices
    print(f"⚠ Camera index {preferred_index} unavailable, scanning indices 0–{SCAN_RANGE - 1} …")
    for i in range(SCAN_RANGE):
        if i == preferred_index:
            continue
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            print(f"✓ Found camera at index {i} (AVFoundation)")
            return cap
        cap.release()

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0 — Camera Capture Baseline")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    args = parser.parse_args()

    requested = {"width": args.width, "height": args.height, "fps": args.fps}

    # --- Open camera ---
    cap = open_camera(args.camera)
    if cap is None:
        print()
        print("✗ No camera found. Checklist:")
        print("  1. Camera permission — System Settings → Privacy & Security → Camera")
        print("     → toggle ON for your terminal app (Terminal / iTerm2 / etc.)")
        print("  2. Continuity Camera (iPhone as webcam):")
        print("     • iPhone and Mac on the same Apple ID")
        print("     • Wi-Fi and Bluetooth enabled on both")
        print("     • iPhone nearby and locked")
        print("     • macOS 13+ / iOS 16+")
        print("  3. Try:  uv run python capture.py --camera 1")
        return

    # Set requested properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Request minimal driver buffer

    actual_w, actual_h, actual_fps = print_capture_props(cap, requested)

    # Rolling windows — ~4 seconds of history at the camera's actual FPS.
    # Use the reported FPS if sane, otherwise fall back to the requested FPS.
    window_size = int((actual_fps if actual_fps > 0 else args.fps) * 4)
    intervals = collections.deque(maxlen=window_size)
    read_times = collections.deque(maxlen=window_size)

    warmup_frames = 30
    frame_count = 0
    prev_time = None

    print(f"\nWarming up ({warmup_frames} frames) …")

    try:
        while True:
            # -- Measure time blocked inside read() --
            t_before_read = time.monotonic()
            ret, frame = cap.read()
            t_after_read = time.monotonic()

            if not ret:
                print("✗ cap.read() returned False — camera disconnected?")
                break

            frame_count += 1

            # -- Discard warmup frames --
            if frame_count <= warmup_frames:
                if frame_count == warmup_frames:
                    print("Warmup complete — measuring.\n")
                prev_time = t_after_read
                continue

            # -- Record timing --
            read_duration = t_after_read - t_before_read
            read_times.append(read_duration)

            if prev_time is not None:
                interval = t_after_read - prev_time
                intervals.append(interval)

            prev_time = t_after_read

            # -- Mirror horizontally --
            frame = cv2.flip(frame, 1)

            # -- Draw overlay --
            if intervals:
                draw_overlay(frame, intervals, read_times, actual_w, actual_h)

            cv2.imshow("Phase 0 — Capture Baseline", frame)

            # -- ESC to exit --
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # -- Summary --
    measured = frame_count - warmup_frames
    if measured > 0 and intervals:
        print("┌─────────────────────────────────────────────┐")
        print("│              Exit Summary                   │")
        print("├─────────────────────────────────────────────┤")
        print(f"│  Frames measured:  {measured:<25}│")
        print(f"│  Interval  p50:    {ms(percentile(intervals, 50)):>7.1f} ms{'':<14}│")
        print(f"│  Interval  p95:    {ms(percentile(intervals, 95)):>7.1f} ms{'':<14}│")
        print(f"│  Interval  p99:    {ms(percentile(intervals, 99)):>7.1f} ms{'':<14}│")
        print(f"│  Read      p50:    {ms(percentile(read_times, 50)):>7.1f} ms{'':<14}│")
        print("└─────────────────────────────────────────────┘")
    else:
        print("No frames measured (exited during warmup).")


if __name__ == "__main__":
    main()
