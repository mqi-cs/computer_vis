"""Phase 1 — Landmark Extraction & Stage Instrumentation.

Builds on Phase 0's capture loop. Adds MediaPipe HandLandmarker in
LIVE_STREAM mode, a 21-point skeleton overlay, per-stage timing, and
JSONL logging. No classification, no gesture logic, no keystrokes.

Usage:
    uv run landmarks.py [--camera INDEX] [--width W] [--height H] [--fps FPS]
                        [--model PATH] [--log PATH] [--headless]
"""

import argparse
import collections
import json
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


# ---------------------------------------------------------------------------
# 21-point hand skeleton connections
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
    # Palm
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]


# ---------------------------------------------------------------------------
# Thread-safe single-slot handoff
# ---------------------------------------------------------------------------

_slot_lock = threading.Lock()
_slot = None  # (result, timestamp_ms, pipeline_ms) or None

# Capture-time registry: timestamp_ms → monotonic capture time.
# Written by main thread before detect_async, popped by callback.
_capture_lock = threading.Lock()
_capture_times = {}


def _on_result(result, output_image, timestamp_ms):
    """MediaPipe callback — stamp arrival, compute pipeline latency, hand off."""
    global _slot
    arrival = time.monotonic()

    with _capture_lock:
        capture_t = _capture_times.pop(timestamp_ms, None)

    pipeline_ms = (arrival - capture_t) * 1000 if capture_t is not None else None

    with _slot_lock:
        _slot = (result, timestamp_ms, pipeline_ms)


def read_slot():
    """Read the latest result. May return None or a previously-seen result."""
    with _slot_lock:
        return _slot


def register_capture_time(timestamp_ms, capture_time):
    """Record when a frame was captured, keyed by its MediaPipe timestamp."""
    with _capture_lock:
        _capture_times[timestamp_ms] = capture_time
        # Safety prune — should never accumulate under normal operation
        if len(_capture_times) > 120:
            for k in sorted(_capture_times)[:60]:
                del _capture_times[k]


# ---------------------------------------------------------------------------
# Monotonic timestamp guard
# ---------------------------------------------------------------------------

_last_ts_ms = -1


def monotonic_timestamp_ms():
    """Integer millisecond timestamp, guaranteed strictly increasing."""
    global _last_ts_ms
    ts_ms = int(time.monotonic() * 1000)
    if ts_ms <= _last_ts_ms:
        ts_ms = _last_ts_ms + 1
    _last_ts_ms = ts_ms
    return ts_ms


# ---------------------------------------------------------------------------
# Helpers (carried from Phase 0)
# ---------------------------------------------------------------------------

def percentile(buf, p):
    """Return the p-th percentile from a deque of floats."""
    if not buf:
        return 0.0
    return float(np.percentile(np.array(buf), p))


def ms(seconds):
    """Seconds → milliseconds, rounded to one decimal."""
    return round(seconds * 1000, 1)


# ---------------------------------------------------------------------------
# Camera (carried from Phase 0)
# ---------------------------------------------------------------------------

SCAN_RANGE = 5


def open_camera(preferred_index):
    """Open a camera with AVFoundation backend, auto-scanning on failure."""
    backend = cv2.CAP_AVFOUNDATION

    cap = cv2.VideoCapture(preferred_index, backend)
    if cap.isOpened():
        print(f"✓ Opened camera index {preferred_index} (AVFoundation)")
        return cap
    cap.release()

    print(f"⚠ Camera index {preferred_index} unavailable, scanning 0–{SCAN_RANGE - 1} …")
    for i in range(SCAN_RANGE):
        if i == preferred_index:
            continue
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            print(f"✓ Found camera at index {i} (AVFoundation)")
            return cap
        cap.release()

    return None


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


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_hand_skeleton(frame, hand_landmarks, w, h, mirror=False):
    """Draw 21-point hand skeleton with connections."""
    points = []
    for lm in hand_landmarks:
        px = int((1.0 - lm.x if mirror else lm.x) * w)
        py = int(lm.y * h)
        points.append((px, py))

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 128), 2, cv2.LINE_AA)

    for px, py in points:
        cv2.circle(frame, (px, py), 4, (255, 0, 128), -1, cv2.LINE_AA)


def draw_overlay(frame, intervals, read_times, detect_call_times,
                 pipeline_times, actual_w, actual_h):
    """Draw live timing stats on the top-left of the frame."""
    p50_int = ms(percentile(intervals, 50))
    p95_int = ms(percentile(intervals, 95))
    p50_read = ms(percentile(read_times, 50))
    p50_det = ms(percentile(detect_call_times, 50))
    equiv_fps = round(1000.0 / p50_int, 1) if p50_int > 0 else 0.0

    lines = [
        f"interval  p50 {p50_int:>6.1f} ms   p95 {p95_int:>6.1f} ms",
        f"read      p50 {p50_read:>6.1f} ms",
        f"detect    p50 {p50_det:>6.1f} ms",
    ]

    if pipeline_times:
        p50_pipe = ms(percentile(pipeline_times, 50))
        p95_pipe = ms(percentile(pipeline_times, 95))
        lines.append(f"pipeline  p50 {p50_pipe:>6.1f} ms   p95 {p95_pipe:>6.1f} ms")

    lines.append(f"FPS (p50)     {equiv_fps:>6.1f}")
    lines.append(f"resolution    {actual_w}x{actual_h}")

    y0 = 30
    for i, line in enumerate(lines):
        y = y0 + i * 28
        cv2.putText(frame, line, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Landmark Extraction & Stage Instrumentation")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", default="hand_landmarker.task",
                        help="Path to hand_landmarker.task model")
    parser.add_argument("--log", default="timing.jsonl",
                        help="JSONL timing log path")
    parser.add_argument("--headless", action="store_true",
                        help="Skip drawing and window display")
    args = parser.parse_args()

    requested = {"width": args.width, "height": args.height, "fps": args.fps}

    # --- Open camera ---
    cap = open_camera(args.camera)
    if cap is None:
        print("\n✗ No camera found. See Phase 0 checklist.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w, actual_h, actual_fps = print_capture_props(cap, requested)

    # --- Create HandLandmarker (LIVE_STREAM) ---
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_on_result,
    )
    landmarker = HandLandmarker.create_from_options(options)
    print("✓ HandLandmarker loaded (LIVE_STREAM, num_hands=1)")

    # --- Rolling windows (~4 seconds) ---
    window_size = int((actual_fps if actual_fps > 0 else args.fps) * 4)
    intervals = collections.deque(maxlen=window_size)
    read_times = collections.deque(maxlen=window_size)
    detect_call_times = collections.deque(maxlen=window_size)
    pipeline_times = collections.deque(maxlen=window_size)

    warmup_frames = 30
    frame_count = 0
    prev_time = None
    last_seen_ts = -1

    log_file = open(args.log, "a")
    print(f"Logging to {args.log}")
    print(f"Warming up ({warmup_frames} frames) …\n")

    try:
        while True:
            # -- Read frame --
            t_before_read = time.monotonic()
            ret, frame = cap.read()
            t_after_read = time.monotonic()

            if not ret:
                print("✗ cap.read() returned False — camera disconnected?")
                break

            frame_count += 1

            # -- Warmup --
            if frame_count <= warmup_frames:
                if frame_count == warmup_frames:
                    print("Warmup complete — measuring.\n")
                prev_time = t_after_read
                continue

            # -- Timing: interval and read --
            read_duration = t_after_read - t_before_read
            read_times.append(read_duration)

            interval = None
            if prev_time is not None:
                interval = t_after_read - prev_time
                intervals.append(interval)
            prev_time = t_after_read

            # -- Submit frame to MediaPipe (async, should not block) --
            ts_ms = monotonic_timestamp_ms()
            register_capture_time(ts_ms, t_after_read)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            t_before_detect = time.monotonic()
            landmarker.detect_async(mp_image, ts_ms)
            t_after_detect = time.monotonic()

            detect_call_duration = t_after_detect - t_before_detect
            detect_call_times.append(detect_call_duration)

            # -- Read latest result from slot --
            slot = read_slot()
            hand_present = False
            pipeline_ms_val = None
            result = None

            if slot is not None:
                result, result_ts, p_ms = slot
                hand_present = bool(result.hand_landmarks)

                # Record pipeline latency only for new results
                if result_ts != last_seen_ts and p_ms is not None:
                    pipeline_times.append(p_ms / 1000)  # store as seconds
                    pipeline_ms_val = round(p_ms, 2)
                    last_seen_ts = result_ts

            # -- JSONL log --
            record = {
                "ts": round(t_after_read, 6),
                "interval_ms": round(interval * 1000, 2) if interval else None,
                "read_ms": round(read_duration * 1000, 2),
                "detect_call_ms": round(detect_call_duration * 1000, 2),
                "pipeline_ms": pipeline_ms_val,
                "hand_present": hand_present,
            }
            log_file.write(json.dumps(record) + "\n")

            # -- Display --
            if not args.headless:
                frame = cv2.flip(frame, 1)

                if hand_present and result is not None:
                    for hand_lms in result.hand_landmarks:
                        draw_hand_skeleton(frame, hand_lms, actual_w, actual_h,
                                           mirror=True)

                if intervals:
                    draw_overlay(frame, intervals, read_times,
                                 detect_call_times, pipeline_times,
                                 actual_w, actual_h)

                cv2.imshow("Phase 1 — Landmarks", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        log_file.close()
        landmarker.close()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    # -- Summary --
    measured = frame_count - warmup_frames
    if measured > 0 and intervals:
        print("┌─────────────────────────────────────────────────────┐")
        print("│                   Exit Summary                     │")
        print("├─────────────────────────────────────────────────────┤")
        print(f"│  Frames measured:    {measured:<30}│")
        print(f"│  Interval    p50:    {ms(percentile(intervals, 50)):>7.1f} ms{'':<20}│")
        print(f"│  Interval    p95:    {ms(percentile(intervals, 95)):>7.1f} ms{'':<20}│")
        print(f"│  Interval    p99:    {ms(percentile(intervals, 99)):>7.1f} ms{'':<20}│")
        print(f"│  Read        p50:    {ms(percentile(read_times, 50)):>7.1f} ms{'':<20}│")
        print(f"│  Detect call p50:    {ms(percentile(detect_call_times, 50)):>7.1f} ms{'':<20}│")
        if pipeline_times:
            print(f"│  Pipeline    p50:    {ms(percentile(pipeline_times, 50)):>7.1f} ms{'':<20}│")
            print(f"│  Pipeline    p95:    {ms(percentile(pipeline_times, 95)):>7.1f} ms{'':<20}│")
            print(f"│  Pipeline    p99:    {ms(percentile(pipeline_times, 99)):>7.1f} ms{'':<20}│")
        print("└─────────────────────────────────────────────────────┘")
    else:
        print("No frames measured (exited during warmup).")


if __name__ == "__main__":
    main()
