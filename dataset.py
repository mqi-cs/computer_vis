"""Phase 4 — Dataset Assembly, Honest Splits & Normalisation.

Loads raw landmark clips from data/, recomputes features via features.py,
builds windowed tensors with shape + motion channels, and provides three
split strategies (random-window, by-clip, by-subject).

No neural network, no PyTorch — numpy and scikit-learn only.

Usage:
    uv run dataset.py [--data-dir ./data] [--seed 42] [--output-dir ./output]
"""

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from features import FEATURE_DIM, ORIGIN_INDEX, extract_features


# ---------------------------------------------------------------------------
# Configuration constants — not magic numbers inline
# ---------------------------------------------------------------------------

WINDOW_LEN = 45        # frames (~1.5s at 30fps)
WINDOW_STRIDE = 5      # frames
MIN_DETECTION_RATE = 0.5   # discard windows below this
MOTION_DIM = 3          # wrist (x, y, z) velocity
FULL_FEATURE_DIM = FEATURE_DIM + MOTION_DIM  # 63 + 3 = 66

GESTURE_LABELS = ["NULL", "ARM", "SWIPE_LEFT", "SWIPE_RIGHT", "PINCH"]
LABEL_TO_IDX = {g: i for i, g in enumerate(GESTURE_LABELS)}
IDX_TO_LABEL = {i: g for g, i in LABEL_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Part A: Clip loading
# ---------------------------------------------------------------------------

def load_clips(data_dir: pathlib.Path) -> List[Dict]:
    """Load all .npz/.json clip pairs from data_dir.

    Returns a list of clip dicts. Skips and reports orphans/corrupt files.
    """
    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        print(f"✗ Data directory {data_dir} does not exist")
        return []

    npz_files = {f.stem: f for f in sorted(data_dir.glob("*.npz"))}
    json_files = {f.stem: f for f in sorted(data_dir.glob("*.json"))}

    # Report orphans
    npz_only = set(npz_files) - set(json_files)
    json_only = set(json_files) - set(npz_files)
    for name in sorted(npz_only):
        print(f"⚠ Orphan .npz (no .json): {name}.npz — skipping")
    for name in sorted(json_only):
        print(f"⚠ Orphan .json (no .npz): {name}.json — skipping")

    paired = sorted(set(npz_files) & set(json_files))
    clips = []

    for name in paired:
        try:
            npz_data = np.load(npz_files[name])
            with open(json_files[name]) as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"⚠ Corrupt file {name}: {e} — skipping")
            continue

        # Validate required arrays
        required = ["landmarks", "world_landmarks", "timestamps", "handedness", "hand_detected"]
        missing = [k for k in required if k not in npz_data]
        if missing:
            print(f"⚠ {name}: missing arrays {missing} — skipping")
            continue

        landmarks = npz_data["landmarks"]
        world_landmarks = npz_data["world_landmarks"]
        timestamps = npz_data["timestamps"]
        handedness = npz_data["handedness"]
        hand_detected = npz_data["hand_detected"]

        # Shape validation
        n_frames = landmarks.shape[0]
        if landmarks.shape != (n_frames, 21, 3):
            print(f"⚠ {name}: landmarks shape {landmarks.shape} — skipping")
            continue
        if world_landmarks.shape != (n_frames, 21, 3):
            print(f"⚠ {name}: world_landmarks shape {world_landmarks.shape} — skipping")
            continue

        gesture = metadata.get("gesture", "?")
        if gesture not in LABEL_TO_IDX:
            print(f"⚠ {name}: unknown gesture '{gesture}' — skipping")
            continue

        clips.append({
            "name": name,
            "gesture": gesture,
            "label_idx": LABEL_TO_IDX[gesture],
            "subject": metadata.get("subject", "unknown"),
            "session": metadata.get("session", "unknown"),
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
            "timestamps": timestamps,
            "handedness": handedness,
            "hand_detected": hand_detected,
            "n_frames": n_frames,
            "metadata": metadata,
        })

    print(f"\nLoaded {len(clips)} clips from {data_dir}")
    return clips


# ---------------------------------------------------------------------------
# Part A: Feature computation (shape + motion channels)
# ---------------------------------------------------------------------------

def build_features_for_clip(clip: Dict) -> np.ndarray:
    """Build per-frame feature vectors for a clip.

    Returns (n_frames, 66) float32 array:
      - columns 0..62: shape channel (Phase 2 normalised, mirrored for left)
      - columns 63..65: motion channel (raw wrist velocity, NOT mirrored)

    Motion channel:
      displacement of WRIST (landmark 0) between consecutive frames,
      divided by actual elapsed time from stored timestamps.
      First frame gets zero motion.

    The motion channel must NOT be mirrored for left hands. Mirroring it
    would swap SWIPE_LEFT and SWIPE_RIGHT directions.
    """
    n = clip["n_frames"]
    landmarks = clip["landmarks"]        # (n, 21, 3)
    handedness = clip["handedness"]       # (n,) str
    hand_detected = clip["hand_detected"] # (n,) bool
    timestamps = clip["timestamps"]       # (n,) int64 ms

    features = np.zeros((n, FULL_FEATURE_DIM), dtype=np.float32)

    # --- Shape channel (63D, mirrored for left) ---
    for i in range(n):
        if hand_detected[i]:
            features[i, :FEATURE_DIM] = extract_features(
                landmarks[i], handedness=str(handedness[i])
            )
        # else: zeros (no-hand sentinel already in place)

    # --- Motion channel (3D, NOT mirrored) ---
    # Use raw wrist landmark position (index 0) for displacement.
    # Divide by elapsed time in seconds from stored timestamps.
    for i in range(1, n):
        if hand_detected[i] and hand_detected[i - 1]:
            dt_ms = float(timestamps[i] - timestamps[i - 1])
            dt_s = dt_ms / 1000.0
            if dt_s > 1e-6:
                # Raw wrist displacement (no mirroring!)
                wrist_curr = landmarks[i, ORIGIN_INDEX]
                wrist_prev = landmarks[i - 1, ORIGIN_INDEX]
                velocity = (wrist_curr - wrist_prev) / dt_s
                features[i, FEATURE_DIM:] = velocity.astype(np.float32)
    # Frame 0 and no-hand frames: motion stays zero

    return features


# ---------------------------------------------------------------------------
# Part A: Windowing & Diagnosis
# ---------------------------------------------------------------------------

# Global diagnosis counters
yield_diag = collections.defaultdict(lambda: {
    "clips": 0, "frames": [], "generated": 0, "surviving": 0,
    "discard_low_det": 0, "discard_other": 0
})

MOTION_ENERGY_THRESHOLD = 0.05  # m/s average motion over the window to be considered "moving"


def window_clip(
    features: np.ndarray,
    hand_detected: np.ndarray,
    clip_gesture: str,
    window_len: int = WINDOW_LEN,
    stride: int = WINDOW_STRIDE,
    min_detection_rate: float = MIN_DETECTION_RATE,
) -> Tuple[List[np.ndarray], List[int]]:
    """Slide a window over per-frame features.
    
    Replaces static detection filter with a motion-energy check.
    Low-motion windows from gesture clips are relabelled to NULL.
    
    Returns:
        windows: List of (window_len, 66) arrays
        labels: List of label indices (may be relabelled to NULL)
    """
    n = features.shape[0]
    windows = []
    labels = []
    
    # Instrumentation
    yield_diag[clip_gesture]["clips"] += 1
    yield_diag[clip_gesture]["frames"].append(n)
    
    null_idx = LABEL_TO_IDX["NULL"]
    clip_idx = LABEL_TO_IDX[clip_gesture]

    for start in range(0, n - window_len + 1, stride):
        end = start + window_len
        yield_diag[clip_gesture]["generated"] += 1
        
        window_feat = features[start:end].copy()
        det_rate = hand_detected[start:end].mean()
        
        # Motion energy: mean of the L2 norm of the velocity vectors (cols 63:66)
        velocities = window_feat[:, FEATURE_DIM:]
        motion_energy = np.linalg.norm(velocities, axis=1).mean()
        
        # Part B, requirement 7: Retain window if motion exceeds threshold OR detection is high enough
        if det_rate < min_detection_rate and motion_energy < MOTION_ENERGY_THRESHOLD:
            yield_diag[clip_gesture]["discard_low_det"] += 1
            continue
            
        # Part B, requirement 8: Relabel low-motion gesture windows to NULL
        # Policy: "Natural lead-in/lead-out frames containing no motion should not be taught
        # as a gesture, to prevent false activations on stationary hands."
        label = clip_idx
        if clip_gesture != "NULL" and motion_energy < MOTION_ENERGY_THRESHOLD:
            label = null_idx
            
        windows.append(window_feat)
        labels.append(label)
        yield_diag[clip_gesture]["surviving"] += 1

    return windows, labels


def print_yield_diagnosis():
    """Print the yield diagnosis table requested in Phase 4.1."""
    print("\n┌────────────────────────────────────────────────────────────────────────────┐")
    print("│                          Window Yield Diagnosis                            │")
    print("├─────────────┬───────┬─────────┬───────────┬─────────────┬─────────┬────────┤")
    print(f"│ {'Class':<11} │ {'Clips':>5} │ {'Mean Fr':>7} │ {'Generated':>9} │ {'Discard Det':>11} │ {'Other':>7} │ {'Survive':>6} │")
    print("├─────────────┼───────┼─────────┼───────────┼─────────────┼─────────┼────────┤")
    
    for g in GESTURE_LABELS:
        d = yield_diag.get(g)
        if not d or d["clips"] == 0:
            continue
        mean_fr = np.mean(d["frames"])
        print(f"│ {g:<11} │ {d['clips']:>5} │ {mean_fr:>7.1f} │ {d['generated']:>9} │ {d['discard_low_det']:>11} │ {d['discard_other']:>7} │ {d['surviving']:>6} │")
    print("└─────────────┴───────┴─────────┴───────────┴─────────────┴─────────┴────────┘")
    
    print("\nConfiguration:")
    print(f"  Window length: {WINDOW_LEN} frames (~{WINDOW_LEN/30:.1f}s)")
    print(f"  Stride: {WINDOW_STRIDE} frames")
    print(f"  Detection threshold: {MIN_DETECTION_RATE}")
    print("  Lead-in policy: Low-motion windows from gesture clips are relabelled to NULL.")
    
    # Check clip length requirement for 15 windows
    req_frames = WINDOW_LEN + (15 - 1) * WINDOW_STRIDE
    print(f"\nYield Analysis:")
    short_clips = False
    for g in GESTURE_LABELS:
        if g == "NULL":
            continue
        d = yield_diag.get(g)
        if d and np.mean(d["frames"]) < req_frames:
            short_clips = True
            
    if short_clips:
        print(f"  ⚠ CAUSE OF LOW YIELD: Clip length is too short.")
        print(f"  Re-recording is required to hit target yield. Minimum clip duration")
        print(f"  needed for 15 windows per clip is {req_frames} frames (~{req_frames/30:.1f}s).")
        print("  Do NOT reduce window length to manufacture windows.")


def build_dataset(
    data_dir: pathlib.Path,
    window_len: int = WINDOW_LEN,
    stride: int = WINDOW_STRIDE,
    min_detection_rate: float = MIN_DETECTION_RATE,
) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
    """Load clips, compute features, window, return arrays."""
    clips = load_clips(data_dir)
    if not clips:
        return clips, np.array([]), np.array([]), np.array([])

    all_windows = []
    all_labels = []
    all_clip_ids = []

    for clip_idx, clip in enumerate(clips):
        features = build_features_for_clip(clip)
        windows, labels = window_clip(
            features, clip["hand_detected"], clip["gesture"],
            window_len, stride, min_detection_rate
        )
        for w, lbl in zip(windows, labels):
            all_windows.append(w)
            all_labels.append(lbl)
            all_clip_ids.append(clip_idx)

    print_yield_diagnosis()

    if not all_windows:
        return clips, np.array([]), np.array([]), np.array([])

    return (
        clips,
        np.array(all_windows, dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
        np.array(all_clip_ids, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Part B: Splitting strategies
# ---------------------------------------------------------------------------

def _split_indices(n: int, ratios: Tuple[float, float, float],
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split indices into train/val/test by ratios."""
    indices = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def split_random_window(
    windows: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deliberately leaky split: pool all windows then split randomly.

    This places overlapping windows (from the same clip) on different sides.
    Included for demonstration of data leakage only.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = _split_indices(len(windows), ratios, rng)

    return (
        windows[train_idx], labels[train_idx],
        windows[val_idx], labels[val_idx],
        windows[test_idx], labels[test_idx],
    )


def split_by_clip(
    clips: List[Dict],
    windows: np.ndarray,
    labels: np.ndarray,
    clip_ids: np.ndarray,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Honest intra-subject split: assign CLIPS to splits, then gather windows.

    No clip's windows appear on more than one side.
    """
    rng = np.random.default_rng(seed)
    n_clips = len(clips)
    train_clips, val_clips, test_clips = _split_indices(n_clips, ratios, rng)

    train_clip_set = set(train_clips)
    val_clip_set = set(val_clips)
    test_clip_set = set(test_clips)

    train_mask = np.isin(clip_ids, list(train_clip_set))
    val_mask = np.isin(clip_ids, list(val_clip_set))
    test_mask = np.isin(clip_ids, list(test_clip_set))

    return (
        windows[train_mask], labels[train_mask],
        windows[val_mask], labels[val_mask],
        windows[test_mask], labels[test_mask],
    )


def split_by_subject(
    clips: List[Dict],
    windows: np.ndarray,
    labels: np.ndarray,
    clip_ids: np.ndarray,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subject-disjoint split: assign SUBJECTS to splits, then gather clips.

    No subject appears on more than one side.
    Falls back to clip split if only one subject.
    """
    subjects = sorted(set(c["subject"] for c in clips))

    if len(subjects) < 3:
        print(f"⚠ Only {len(subjects)} subject(s): {subjects}")
        print("  Subject split requires ≥3 subjects. Falling back to clip split.")
        return split_by_clip(clips, windows, labels, clip_ids, seed, ratios)

    rng = np.random.default_rng(seed)
    subj_indices = rng.permutation(len(subjects))
    n_train = max(1, int(len(subjects) * ratios[0]))
    n_val = max(1, int(len(subjects) * ratios[1]))

    train_subjects = set(subjects[i] for i in subj_indices[:n_train])
    val_subjects = set(subjects[i] for i in subj_indices[n_train:n_train + n_val])
    test_subjects = set(subjects[i] for i in subj_indices[n_train + n_val:])

    # Map clip_ids to subject sets
    clip_subject = np.array([clips[cid]["subject"] for cid in clip_ids])
    train_mask = np.isin(clip_subject, list(train_subjects))
    val_mask = np.isin(clip_subject, list(val_subjects))
    test_mask = np.isin(clip_subject, list(test_subjects))

    print(f"  Subject split: train={train_subjects}, val={val_subjects}, test={test_subjects}")

    return (
        windows[train_mask], labels[train_mask],
        windows[val_mask], labels[val_mask],
        windows[test_mask], labels[test_mask],
    )


# ---------------------------------------------------------------------------
# Part D: Normalisation statistics
# ---------------------------------------------------------------------------

def compute_norm_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from training data ONLY.

    Args:
        X_train: (N, window_len, feature_dim) float32

    Returns:
        mean: (feature_dim,) float32
        std: (feature_dim,) float32
    """
    # Flatten across windows and time steps for per-feature stats
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray,
               eps: float = 1e-8) -> np.ndarray:
    """Standardise features using pre-computed mean and std.

    Args:
        X: (..., feature_dim) array
        mean: (feature_dim,) array
        std: (feature_dim,) array

    Returns normalised copy.
    """
    return ((X - mean) / (std + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset report
# ---------------------------------------------------------------------------

def print_dataset_report(clips: List[Dict], all_labels: np.ndarray,
                         all_clip_ids: np.ndarray):
    """Print comprehensive dataset summary."""
    gesture_stats = collections.defaultdict(
        lambda: {"clips": set(), "windows": 0, "subjects": set()}
    )
    total_frames = 0
    total_duration = 0.0

    for i, clip in enumerate(clips):
        g = clip["gesture"]
        gesture_stats[g]["clips"].add(i)
        gesture_stats[g]["subjects"].add(clip["subject"])
        total_frames += clip["n_frames"]
        total_duration += clip["n_frames"] / 30.0  # approx

    # Count windows per gesture
    for w_idx in range(len(all_labels)):
        g = IDX_TO_LABEL[all_labels[w_idx]]
        gesture_stats[g]["windows"] += 1

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│                     Dataset Report                         │")
    print("├──────────────┬───────┬─────────┬───────────┬───────────────┤")
    print(f"│ {'Gesture':<12} │ {'Clips':>5} │ {'Windows':>7} │ {'Subjects':>9} │ {'Balance':>13} │")
    print("├──────────────┼───────┼─────────┼───────────┼───────────────┤")

    total_windows = len(all_labels)
    for g in GESTURE_LABELS:
        s = gesture_stats.get(g)
        if s is None:
            continue
        n_clips = len(s["clips"])
        n_windows = s["windows"]
        n_subjects = len(s["subjects"])
        pct = (n_windows / total_windows * 100) if total_windows > 0 else 0
        print(f"│ {g:<12} │ {n_clips:>5} │ {n_windows:>7} │ {n_subjects:>9} │ {pct:>11.1f}% │")

    print("├──────────────┼───────┼─────────┼───────────┼───────────────┤")
    print(f"│ {'TOTAL':<12} │ {len(clips):>5} │ {total_windows:>7} │ {'':>9} │ {total_duration / 60:>10.1f} min │")
    print("└──────────────┴───────┴─────────┴───────────┴───────────────┘")


# ---------------------------------------------------------------------------
# Save/load splits
# ---------------------------------------------------------------------------

def save_split(output_dir: pathlib.Path, split_name: str,
               X_train, y_train, X_val, y_val, X_test, y_test,
               mean, std):
    """Save split arrays and normalisation stats to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"split_{split_name}.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    np.savez(
        output_dir / "norm_stats.npz",
        mean=mean, std=std,
    )
    print(f"  Saved split_{split_name}.npz and norm_stats.npz to {output_dir}")


def load_split(output_dir: pathlib.Path, split_name: str):
    """Load a saved split and norm stats."""
    split_data = np.load(output_dir / f"split_{split_name}.npz")
    norm_data = np.load(output_dir / "norm_stats.npz")
    return (
        split_data["X_train"], split_data["y_train"],
        split_data["X_val"], split_data["y_val"],
        split_data["X_test"], split_data["y_test"],
        norm_data["mean"], norm_data["std"],
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Dataset Assembly & Splits")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-len", type=int, default=WINDOW_LEN)
    parser.add_argument("--stride", type=int, default=WINDOW_STRIDE)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and window ---
    clips, windows, labels, clip_ids = build_dataset(
        data_dir, args.window_len, args.stride
    )
    if len(windows) == 0:
        print("✗ No windows generated. Check data directory.")
        return

    print_dataset_report(clips, labels, clip_ids)

    # --- Splits ---
    print("\n--- Split: random_window (deliberately leaky) ---")
    rw = split_random_window(windows, labels, args.seed)
    X_tr, y_tr, X_va, y_va, X_te, y_te = rw
    print(f"  train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")

    mean, std = compute_norm_stats(X_tr)
    save_split(output_dir, "random_window", *rw, mean, std)

    print("\n--- Split: by_clip ---")
    bc = split_by_clip(clips, windows, labels, clip_ids, args.seed)
    X_tr, y_tr, X_va, y_va, X_te, y_te = bc
    print(f"  train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")

    mean, std = compute_norm_stats(X_tr)
    save_split(output_dir, "by_clip", *bc, mean, std)

    print("\n--- Split: by_subject ---")
    bs = split_by_subject(clips, windows, labels, clip_ids, args.seed)
    X_tr, y_tr, X_va, y_va, X_te, y_te = bs
    print(f"  train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")

    mean_bs, std_bs = compute_norm_stats(X_tr)
    save_split(output_dir, "by_subject", *bs, mean_bs, std_bs)

    print("\n✓ Dataset assembly complete.")


if __name__ == "__main__":
    main()
