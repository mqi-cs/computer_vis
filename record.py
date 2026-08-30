"""Phase 3 — Data Collection Tool.

Guided clip recorder that captures raw 21-point hand landmarks (normalised
and world coordinates), timestamps, and handedness into .npz files with
.json sidecar metadata. No feature computation, no model code.

Usage:
    uv run record.py --gesture PINCH --clips 10 --subject s01
    uv run record.py --gesture NULL --null-mode --null-minutes 5

Controls during review:
    SPACE   keep clip and advance
    R       redo clip (discards the recording, no partial file on disk)
    ESC     abort session cleanly
"""

import argparse
import collections
import datetime
import json
import os
import pathlib
import tempfile
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


TOOL_VERSION = "v0.3.0"

GESTURE_SET = ["NULL", "ARM", "SWIPE_LEFT", "SWIPE_RIGHT", "PINCH"]

# Data collection protocol prompts — cycled on screen during countdown
PROTOCOL_PROMPTS = [
    "Vary SPEED: try fast, medium, slow",
    "Vary DISTANCE: move hand closer / further",
    "Include natural lead-in and lead-out",
    "Try BOTH hands across clips",
    "Keep hand visible throughout",
]

MIN_DETECTION_RATE_DEFAULT = 0.7


# ---------------------------------------------------------------------------
# Thread-safe single-slot handoff (same pattern as landmarks.py)
# ---------------------------------------------------------------------------

_slot_lock = threading.Lock()
_slot = None

_capture_lock = threading.Lock()
_capture_times = {}

_last_ts_ms = -1


def _on_result(result, output_image, timestamp_ms):
    """MediaPipe callback — minimal work, hand off to slot."""
    global _slot
    arrival = time.monotonic()
    with _capture_lock:
        capture_t = _capture_times.pop(timestamp_ms, None)
    pipeline_ms = (arrival - capture_t) * 1000 if capture_t is not None else None
    with _slot_lock:
        _slot = (result, timestamp_ms, pipeline_ms)


def read_slot():
    with _slot_lock:
        return _slot


def register_capture_time(timestamp_ms, capture_time):
    with _capture_lock:
        _capture_times[timestamp_ms] = capture_time
        if len(_capture_times) > 120:
            for k in sorted(_capture_times)[:60]:
                del _capture_times[k]


def monotonic_timestamp_ms():
    global _last_ts_ms
    ts_ms = int(time.monotonic() * 1000)
    if ts_ms <= _last_ts_ms:
        ts_ms = _last_ts_ms + 1
    _last_ts_ms = ts_ms
    return ts_ms


# ---------------------------------------------------------------------------
# Camera helpers (carried from landmarks.py)
# ---------------------------------------------------------------------------

SCAN_RANGE = 5


def open_camera(preferred_index):
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


# ---------------------------------------------------------------------------
# File I/O — atomic writes
# ---------------------------------------------------------------------------

def next_clip_index(data_dir, gesture, subject, session):
    """Find the next available clip index for the given naming prefix."""
    prefix = f"{gesture}__{subject}__{session}__"
    existing = []
    if data_dir.exists():
        for f in data_dir.iterdir():
            if f.name.startswith(prefix) and f.suffix == ".npz":
                try:
                    idx_str = f.stem.split("__")[-1]
                    existing.append(int(idx_str))
                except ValueError:
                    continue
    return max(existing, default=-1) + 1


def clip_filename(gesture, subject, session, index):
    """Return base filename without extension."""
    return f"{gesture}__{subject}__{session}__{index:04d}"


def save_clip_atomic(data_dir, basename, arrays_dict, metadata_dict):
    """Write .npz and .json atomically via temp files + rename.

    Returns the paths of the saved files or raises on failure.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    npz_path = data_dir / f"{basename}.npz"
    json_path = data_dir / f"{basename}.json"

    # Write .npz atomically
    fd_npz, tmp_npz = tempfile.mkstemp(suffix=".tmp.npz", dir=data_dir)
    os.close(fd_npz)
    try:
        np.savez_compressed(tmp_npz, **arrays_dict)
        os.rename(tmp_npz, npz_path)
    except BaseException:
        if os.path.exists(tmp_npz):
            os.remove(tmp_npz)
        raise

    # Write .json atomically
    fd_json, tmp_json = tempfile.mkstemp(suffix=".tmp.json", dir=data_dir)
    os.close(fd_json)
    try:
        with open(tmp_json, "w") as f:
            json.dump(metadata_dict, f, indent=2)
        os.rename(tmp_json, json_path)
    except BaseException:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)
        # Also remove the already-written .npz to avoid orphans
        if npz_path.exists():
            npz_path.unlink()
        raise

    return npz_path, json_path


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def print_dataset_summary(data_dir):
    """Scan data/ and print clips per gesture, per subject, total minutes."""
    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        print("No data directory found.")
        return

    gesture_stats = collections.defaultdict(lambda: {"clips": 0, "frames": 0, "detected": 0, "duration": 0.0})
    subject_clips = collections.defaultdict(int)

    for json_file in sorted(data_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        gesture = meta.get("gesture", "?")
        subject = meta.get("subject", "?")
        frames = meta.get("frame_count", 0)
        detected = meta.get("hand_detected_count", 0)
        duration = meta.get("duration_sec", 0.0)

        gesture_stats[gesture]["clips"] += 1
        gesture_stats[gesture]["frames"] += frames
        gesture_stats[gesture]["detected"] += detected
        gesture_stats[gesture]["duration"] += duration
        subject_clips[subject] += 1

    if not gesture_stats:
        print("No clips found in data directory.")
        return

    total_clips = sum(s["clips"] for s in gesture_stats.values())
    total_minutes = sum(s["duration"] for s in gesture_stats.values()) / 60

    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│                    Dataset Summary                      │")
    print("├──────────────┬───────┬────────┬───────────┬─────────────┤")
    print(f"│ {'Gesture':<12} │ {'Clips':>5} │ {'Frames':>6} │ {'Detect %':>9} │ {'Duration':>11} │")
    print("├──────────────┼───────┼────────┼───────────┼─────────────┤")
    for gesture in GESTURE_SET:
        s = gesture_stats.get(gesture)
        if s is None:
            continue
        det_rate = (s["detected"] / s["frames"] * 100) if s["frames"] > 0 else 0
        dur_str = f"{s['duration']:.1f}s"
        print(f"│ {gesture:<12} │ {s['clips']:>5} │ {s['frames']:>6} │ {det_rate:>8.1f}% │ {dur_str:>11} │")
    print("├──────────────┼───────┼────────┼───────────┼─────────────┤")
    print(f"│ {'TOTAL':<12} │ {total_clips:>5} │ {'':>6} │ {'':>9} │ {total_minutes:>8.1f} min │")
    print("└──────────────┴───────┴────────┴───────────┴─────────────┘")

    if subject_clips:
        print(f"\nSubjects: {dict(subject_clips)}")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]


def draw_skeleton(frame, hand_landmarks, w, h, mirror=False):
    """Draw hand skeleton on frame."""
    points = []
    for lm in hand_landmarks:
        px = int((1.0 - lm.x if mirror else lm.x) * w)
        py = int(lm.y * h)
        points.append((px, py))
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 128), 2, cv2.LINE_AA)
    for px, py in points:
        cv2.circle(frame, (px, py), 4, (255, 0, 128), -1, cv2.LINE_AA)


def draw_text(frame, text, pos, scale=0.7, color=(0, 255, 0), thickness=1):
    """Draw text with shadow for readability."""
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_big_text(frame, text, color=(0, 0, 255)):
    """Draw large centered text on frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 3.0
    thickness = 5
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Frame capture helper
# ---------------------------------------------------------------------------

def capture_frame_data(cap, landmarker):
    """Read one frame, submit to MediaPipe, return frame and slot data.

    Returns (frame, ts_ms, slot) where slot may be None or stale.
    Does NOT block on inference.
    """
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    ts_ms = monotonic_timestamp_ms()
    register_capture_time(ts_ms, time.monotonic())

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    landmarker.detect_async(mp_image, ts_ms)

    slot = read_slot()
    return frame, ts_ms, slot


def extract_slot_data(slot):
    """Extract landmarks, world landmarks, handedness from a slot result.

    Returns (hand_present, landmarks_21x3, world_landmarks_21x3, handedness_str).
    """
    if slot is None:
        return False, np.zeros((21, 3), dtype=np.float32), np.zeros((21, 3), dtype=np.float32), "None"

    result, _, _ = slot
    if result is None or not result.hand_landmarks:
        return False, np.zeros((21, 3), dtype=np.float32), np.zeros((21, 3), dtype=np.float32), "None"

    # Normalised image-space landmarks
    lms = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
                    dtype=np.float32)

    # World landmarks (3D in metres)
    if result.hand_world_landmarks:
        wlms = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]],
                         dtype=np.float32)
    else:
        wlms = np.zeros((21, 3), dtype=np.float32)

    # Handedness
    handedness_str = "None"
    if result.handedness and result.handedness[0]:
        handedness_str = result.handedness[0][0].category_name

    return True, lms, wlms, handedness_str


# ---------------------------------------------------------------------------
# Recording states
# ---------------------------------------------------------------------------

STATE_COUNTDOWN = "countdown"
STATE_RECORDING = "recording"
STATE_REVIEW = "review"
STATE_DONE = "done"
STATE_ABORTED = "aborted"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Data Collection Tool")
    parser.add_argument("--gesture", required=True, choices=GESTURE_SET,
                        help="Gesture class to record")
    parser.add_argument("--subject", default="subject_01", help="Subject identifier")
    parser.add_argument("--session", default="s01", help="Session identifier")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Clip duration in seconds (default: 2.0)")
    parser.add_argument("--clips", type=int, default=10,
                        help="Number of clips to record (default: 10)")
    parser.add_argument("--hand-used", choices=["right", "left", "both"], default="right")
    parser.add_argument("--lighting", choices=["bright", "normal", "dim"], default="normal")
    parser.add_argument("--distance", choices=["near", "normal", "far"], default="normal")
    parser.add_argument("--min-detection-rate", type=float, default=MIN_DETECTION_RATE_DEFAULT,
                        help=f"Minimum detection rate to accept a clip (default: {MIN_DETECTION_RATE_DEFAULT})")
    parser.add_argument("--null-mode", action="store_true",
                        help="Long-form continuous NULL recording mode")
    parser.add_argument("--null-minutes", type=float, default=5.0,
                        help="Duration for null-mode in minutes (default: 5.0)")
    parser.add_argument("--data-dir", default="./data", help="Data output directory")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", default="hand_landmarker.task")
    args = parser.parse_args()

    if args.null_mode and args.gesture != "NULL":
        print("✗ --null-mode requires --gesture NULL")
        return

    data_dir = pathlib.Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Open camera ---
    cap = open_camera(args.camera)
    if cap is None:
        print("\n✗ No camera found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {actual_w}x{actual_h}")

    # --- Create HandLandmarker ---
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
    print("✓ HandLandmarker loaded")

    # Warmup
    print("Warming up (30 frames) …")
    for _ in range(30):
        capture_frame_data(cap, landmarker)
    print("Warmup complete.\n")

    if args.null_mode:
        _record_null_longform(cap, landmarker, args, data_dir, actual_w, actual_h)
    else:
        _record_clips(cap, landmarker, args, data_dir, actual_w, actual_h)

    # Cleanup
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    # Dataset summary
    print_dataset_summary(data_dir)


# ---------------------------------------------------------------------------
# Clip recording mode
# ---------------------------------------------------------------------------

def _record_clips(cap, landmarker, args, data_dir, actual_w, actual_h):
    """Guided clip recording with countdown → record → review loop."""
    clip_index = next_clip_index(data_dir, args.gesture, args.subject, args.session)
    clips_saved = 0
    prompt_idx = 0
    prev_detection_rate = 1.0

    print(f"Recording {args.clips} clips of '{args.gesture}' "
          f"({args.duration}s each), starting at index {clip_index}")
    print("Controls: SPACE=keep, R=redo, ESC=abort\n")

    while clips_saved < args.clips:
        state = STATE_COUNTDOWN
        countdown_start = time.monotonic()
        countdown_duration = 3.0
        record_start = None

        # Clip buffers (in-memory, written to disk at clip end)
        buf_landmarks = []
        buf_world_landmarks = []
        buf_timestamps = []
        buf_handedness = []
        buf_detected = []

        save_pending = False
        review_start = None

        while True:
            frame, ts_ms, slot = capture_frame_data(cap, landmarker)
            if frame is None:
                print("✗ Camera disconnected")
                return

            display = cv2.flip(frame, 1)
            hand_present, lms, wlms, hand_str = extract_slot_data(slot)

            # Draw skeleton when visible
            if hand_present and slot is not None:
                result, _, _ = slot
                if result and result.hand_landmarks:
                    for hand_lms in result.hand_landmarks:
                        draw_skeleton(display, hand_lms, actual_w, actual_h, mirror=True)

            # Status bar
            draw_text(display, f"[{args.gesture}] clip {clips_saved + 1}/{args.clips}  "
                       f"subject={args.subject}  session={args.session}",
                       (10, 30), scale=0.55, color=(200, 200, 200))

            elapsed_mono = time.monotonic()

            # ---- STATE: COUNTDOWN ----
            if state == STATE_COUNTDOWN:
                remaining = countdown_duration - (elapsed_mono - countdown_start)
                if remaining <= 0:
                    state = STATE_RECORDING
                    record_start = time.monotonic()
                    continue

                count_num = int(remaining) + 1
                draw_big_text(display, str(count_num), color=(0, 200, 255))

                # Detection rate warning from previous clip
                if prev_detection_rate < args.min_detection_rate:
                    draw_text(display, f"!! Low detection rate: {prev_detection_rate:.0%} "
                               "— check lighting / framing !!",
                               (10, actual_h - 60), scale=0.6, color=(0, 0, 255))

                # Protocol prompt
                prompt = PROTOCOL_PROMPTS[prompt_idx % len(PROTOCOL_PROMPTS)]
                draw_text(display, prompt, (10, actual_h - 25), scale=0.55, color=(255, 255, 100))

            # ---- STATE: RECORDING ----
            elif state == STATE_RECORDING:
                elapsed_rec = elapsed_mono - record_start

                # Buffer frame data
                buf_landmarks.append(lms)
                buf_world_landmarks.append(wlms)
                buf_timestamps.append(ts_ms)
                buf_handedness.append(hand_str)
                buf_detected.append(hand_present)

                if elapsed_rec >= args.duration:
                    state = STATE_REVIEW
                    review_start = time.monotonic()
                    save_pending = True

                    # Compute detection rate
                    n_detected = sum(buf_detected)
                    n_frames = len(buf_detected)
                    det_rate = n_detected / n_frames if n_frames > 0 else 0.0
                    prev_detection_rate = det_rate
                    continue

                # Recording indicator
                progress = f"{elapsed_rec:.1f}s / {args.duration:.1f}s"
                draw_text(display, f"  RECORDING  {progress}",
                           (10, 70), scale=0.7, color=(0, 0, 255), thickness=2)

                # Pulsing red circle
                if int(elapsed_rec * 4) % 2 == 0:
                    cv2.circle(display, (25, 95), 10, (0, 0, 255), -1)

            # ---- STATE: REVIEW ----
            elif state == STATE_REVIEW:
                n_frames = len(buf_detected)
                n_detected = sum(buf_detected)
                det_rate = n_detected / n_frames if n_frames > 0 else 0.0

                det_color = (0, 255, 0) if det_rate >= args.min_detection_rate else (0, 0, 255)

                draw_text(display, f"Clip recorded: {n_frames} frames, "
                           f"detection {det_rate:.0%}",
                           (10, 70), scale=0.6, color=det_color)

                if det_rate < args.min_detection_rate:
                    draw_text(display, "!! POOR DETECTION — consider redo (R) !!",
                               (10, 100), scale=0.6, color=(0, 0, 255))

                draw_text(display, "SPACE=save  R=redo  ESC=abort",
                           (10, actual_h - 25), scale=0.55, color=(200, 200, 200))

                # Auto-advance after 1.5 seconds
                if save_pending and (elapsed_mono - review_start) > 1.5:
                    # Auto-save
                    basename = clip_filename(args.gesture, args.subject,
                                             args.session, clip_index)
                    _save_clip_from_buffers(
                        data_dir, basename, args,
                        buf_landmarks, buf_world_landmarks,
                        buf_timestamps, buf_handedness, buf_detected,
                    )
                    print(f"  ✓ Saved {basename} ({n_frames} frames, {det_rate:.0%} detect)")

                    clips_saved += 1
                    clip_index += 1
                    prompt_idx += 1
                    save_pending = False
                    break  # next clip

            cv2.imshow("Record", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("\n⚠ Session aborted by user.")
                return
            elif key == ord(" ") and state == STATE_REVIEW:
                # Manual save
                basename = clip_filename(args.gesture, args.subject,
                                         args.session, clip_index)
                n_frames = len(buf_detected)
                det_rate = sum(buf_detected) / n_frames if n_frames > 0 else 0.0
                _save_clip_from_buffers(
                    data_dir, basename, args,
                    buf_landmarks, buf_world_landmarks,
                    buf_timestamps, buf_handedness, buf_detected,
                )
                print(f"  ✓ Saved {basename} ({n_frames} frames, {det_rate:.0%} detect)")

                clips_saved += 1
                clip_index += 1
                prompt_idx += 1
                save_pending = False
                break
            elif key == ord("r") and state == STATE_REVIEW:
                # Redo — discard buffers, no file written
                print(f"  ↻ Redo clip {clips_saved + 1}")
                save_pending = False
                break  # restart this clip

    print(f"\n✓ Session complete: {clips_saved} clips saved.")


# ---------------------------------------------------------------------------
# NULL long-form recording mode
# ---------------------------------------------------------------------------

def _record_null_longform(cap, landmarker, args, data_dir, actual_w, actual_h):
    """Continuous recording for NULL class — no interaction needed."""
    clip_index = next_clip_index(data_dir, args.gesture, args.subject, args.session)
    total_duration = args.null_minutes * 60

    print(f"NULL long-form mode: recording {args.null_minutes:.1f} min continuously")
    print("Press ESC to stop early.\n")

    buf_landmarks = []
    buf_world_landmarks = []
    buf_timestamps = []
    buf_handedness = []
    buf_detected = []

    record_start = time.monotonic()

    try:
        while True:
            frame, ts_ms, slot = capture_frame_data(cap, landmarker)
            if frame is None:
                print("✗ Camera disconnected")
                break

            hand_present, lms, wlms, hand_str = extract_slot_data(slot)

            buf_landmarks.append(lms)
            buf_world_landmarks.append(wlms)
            buf_timestamps.append(ts_ms)
            buf_handedness.append(hand_str)
            buf_detected.append(hand_present)

            elapsed = time.monotonic() - record_start

            # Display
            display = cv2.flip(frame, 1)

            if hand_present and slot is not None:
                result, _, _ = slot
                if result and result.hand_landmarks:
                    for hand_lms in result.hand_landmarks:
                        draw_skeleton(display, hand_lms, actual_w, actual_h, mirror=True)

            remaining = total_duration - elapsed
            draw_text(display, f"NULL RECORDING  {elapsed:.0f}s / {total_duration:.0f}s  "
                       f"({remaining:.0f}s left)",
                       (10, 30), scale=0.6, color=(0, 0, 255))

            frames_so_far = len(buf_detected)
            det_so_far = sum(buf_detected)
            det_rate = det_so_far / frames_so_far if frames_so_far > 0 else 0.0
            draw_text(display, f"frames: {frames_so_far}  detect: {det_rate:.0%}",
                       (10, 60), scale=0.55, color=(200, 200, 200))

            cv2.imshow("Record", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("\nStopped early by user.")
                break

            if elapsed >= total_duration:
                print("\nDuration reached.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # Save
    n_frames = len(buf_detected)
    if n_frames > 0:
        basename = clip_filename(args.gesture, args.subject, args.session, clip_index)
        det_rate = sum(buf_detected) / n_frames
        _save_clip_from_buffers(
            data_dir, basename, args,
            buf_landmarks, buf_world_landmarks,
            buf_timestamps, buf_handedness, buf_detected,
        )
        actual_dur = n_frames / args.fps if args.fps > 0 else 0
        print(f"✓ Saved {basename} ({n_frames} frames, {actual_dur:.1f}s, {det_rate:.0%} detect)")
    else:
        print("No frames captured.")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_clip_from_buffers(data_dir, basename, args,
                            buf_landmarks, buf_world_landmarks,
                            buf_timestamps, buf_handedness, buf_detected):
    """Package buffers into .npz + .json and write atomically."""
    n_frames = len(buf_detected)
    n_detected = sum(buf_detected)
    det_rate = n_detected / n_frames if n_frames > 0 else 0.0

    arrays = {
        "landmarks": np.array(buf_landmarks, dtype=np.float32),         # (n, 21, 3)
        "world_landmarks": np.array(buf_world_landmarks, dtype=np.float32),  # (n, 21, 3)
        "timestamps": np.array(buf_timestamps, dtype=np.int64),          # (n,)
        "handedness": np.array(buf_handedness, dtype="<U10"),             # (n,)
        "hand_detected": np.array(buf_detected, dtype=bool),             # (n,)
    }

    metadata = {
        "gesture": args.gesture,
        "subject": args.subject,
        "session": args.session,
        "hand_used": args.hand_used,
        "lighting": args.lighting,
        "camera_distance": args.distance,
        "duration_sec": round(n_frames / args.fps, 2) if args.fps > 0 else 0.0,
        "frame_count": n_frames,
        "hand_detected_count": n_detected,
        "detection_rate": round(det_rate, 4),
        "tool_version": TOOL_VERSION,
        "utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "filename_npz": f"{basename}.npz",
    }

    save_clip_atomic(data_dir, basename, arrays, metadata)


if __name__ == "__main__":
    main()
