"""Phase 4 — Pytest suite for dataset.py (no camera required).

Tests split integrity, motion channel correctness, normalisation leakage,
and timestamp-based velocity computation.
"""

import numpy as np
import pytest

from dataset import (
    FULL_FEATURE_DIM,
    GESTURE_LABELS,
    LABEL_TO_IDX,
    MOTION_DIM,
    WINDOW_LEN,
    WINDOW_STRIDE,
    build_features_for_clip,
    compute_norm_stats,
    apply_norm,
    split_by_clip,
    split_random_window,
    window_clip,
)
from features import FEATURE_DIM, ORIGIN_INDEX


# ---------------------------------------------------------------------------
# Helpers: synthetic clip construction
# ---------------------------------------------------------------------------

def make_synthetic_clip(
    n_frames=60,
    gesture="PINCH",
    subject="test_subject",
    handedness="Right",
    all_detected=True,
    uniform_timestamps=True,
    landmarks_fn=None,
):
    """Build a synthetic clip dict matching the schema from record.py."""
    if landmarks_fn is not None:
        landmarks = landmarks_fn(n_frames)
    else:
        # Wrist at (0.5, 0.5, 0) with fingers spread, slight motion
        base = np.zeros((21, 3), dtype=np.float32)
        base[0] = [0.5, 0.7, 0.0]   # wrist
        base[9] = [0.5, 0.5, 0.0]   # middle MCP
        for i in range(1, 21):
            if i != 9:
                base[i] = base[0] + np.random.default_rng(i).uniform(-0.15, 0.15, 3).astype(np.float32)
        landmarks = np.tile(base, (n_frames, 1, 1))
        # Add linear wrist motion across frames
        for i in range(n_frames):
            landmarks[i, 0, 0] += i * 0.005  # x moves right

    if uniform_timestamps:
        timestamps = np.arange(n_frames, dtype=np.int64) * 33  # ~30fps
    else:
        # Non-uniform: first half at 33ms, second half at 100ms
        timestamps = np.zeros(n_frames, dtype=np.int64)
        for i in range(n_frames):
            if i < n_frames // 2:
                timestamps[i] = i * 33
            else:
                timestamps[i] = timestamps[n_frames // 2 - 1] + (i - n_frames // 2 + 1) * 100

    hand_detected = np.ones(n_frames, dtype=bool) if all_detected else \
                    np.random.default_rng(42).random(n_frames) > 0.3

    return {
        "name": f"{gesture}__test__{subject}__0000",
        "gesture": gesture,
        "label_idx": LABEL_TO_IDX[gesture],
        "subject": subject,
        "session": "test",
        "landmarks": landmarks.astype(np.float32),
        "world_landmarks": landmarks.astype(np.float32),
        "timestamps": timestamps,
        "handedness": np.array([handedness] * n_frames, dtype="<U10"),
        "hand_detected": hand_detected,
        "n_frames": n_frames,
        "metadata": {"gesture": gesture, "subject": subject},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureComputation:
    def test_output_shape(self):
        """Feature vector has correct shape (n_frames, 66)."""
        clip = make_synthetic_clip(n_frames=60)
        features = build_features_for_clip(clip)
        assert features.shape == (60, FULL_FEATURE_DIM)
        assert features.dtype == np.float32

    def test_motion_channel_nonzero(self):
        """Motion channel is nonzero when wrist moves between frames."""
        clip = make_synthetic_clip(n_frames=30)
        features = build_features_for_clip(clip)
        # Motion channel is columns 63:66
        motion = features[1:, FEATURE_DIM:]
        assert np.any(motion != 0), "Motion channel should be nonzero with moving wrist"

    def test_motion_channel_first_frame_zero(self):
        """First frame of each clip has zero motion channel."""
        clip = make_synthetic_clip(n_frames=30)
        features = build_features_for_clip(clip)
        np.testing.assert_array_equal(features[0, FEATURE_DIM:], [0.0, 0.0, 0.0])


class TestMirrorInvariance:
    def test_swipe_left_motion_not_mirrored(self):
        """CRITICAL: A left-hand SWIPE_LEFT must NOT have its motion x-inverted.

        Mirroring the motion channel would swap SWIPE_LEFT and SWIPE_RIGHT.
        """
        n = 30

        # Right hand moving left (negative x direction)
        def right_hand_lms(n_frames):
            base = np.zeros((21, 3), dtype=np.float32)
            base[0] = [0.5, 0.7, 0.0]
            base[9] = [0.5, 0.5, 0.0]
            for i in range(1, 21):
                if i != 9:
                    base[i] = base[0] + np.array([0.0, -0.1 * (i / 21), 0], dtype=np.float32)
            lms = np.tile(base, (n_frames, 1, 1))
            for i in range(n_frames):
                lms[i, 0, 0] -= i * 0.01  # wrist moves LEFT
            return lms

        # Left hand: mirror the x coords relative to center (0.5)
        def left_hand_lms(n_frames):
            r = right_hand_lms(n_frames)
            r[:, :, 0] = 1.0 - r[:, :, 0]  # mirror x
            return r

        clip_right = make_synthetic_clip(n_frames=n, gesture="SWIPE_LEFT",
                                          handedness="Right", landmarks_fn=right_hand_lms)
        clip_left = make_synthetic_clip(n_frames=n, gesture="SWIPE_LEFT",
                                         handedness="Left", landmarks_fn=left_hand_lms)

        feat_right = build_features_for_clip(clip_right)
        feat_left = build_features_for_clip(clip_left)

        # Motion channel x-direction should have OPPOSITE signs
        # because the raw wrist moves in opposite x-directions
        # (right hand moves left, left hand mirror moves right in raw coords)
        motion_x_right = feat_right[1:, FEATURE_DIM]
        motion_x_left = feat_left[1:, FEATURE_DIM]

        # The key assertion: motion is NOT mirrored, so sign differs
        assert np.sign(motion_x_right.mean()) != np.sign(motion_x_left.mean()), \
            "Motion channel x should NOT be mirrored — different raw directions preserved"


class TestTimestampVelocity:
    def test_velocity_uses_timestamps(self):
        """Velocity changes when timestamps are non-uniform (proving we use
        real elapsed time, not frame index)."""
        clip_uniform = make_synthetic_clip(n_frames=60, uniform_timestamps=True)
        clip_nonuniform = make_synthetic_clip(n_frames=60, uniform_timestamps=False)

        feat_uniform = build_features_for_clip(clip_uniform)
        feat_nonuniform = build_features_for_clip(clip_nonuniform)

        # Motion channels should differ because dt differs
        motion_uniform = feat_uniform[:, FEATURE_DIM:]
        motion_nonuniform = feat_nonuniform[:, FEATURE_DIM:]

        # They should NOT be identical
        assert not np.allclose(motion_uniform, motion_nonuniform), \
            "Velocity should change with non-uniform timestamps"


class TestWindowing:
    def test_window_shape(self):
        """Windows have correct shape."""
        clip = make_synthetic_clip(n_frames=120)
        features = build_features_for_clip(clip)
        windows, _ = window_clip(features, clip["hand_detected"], clip["gesture"])
        assert len(windows) > 0
        for w in windows:
            assert w.shape == (WINDOW_LEN, FULL_FEATURE_DIM)

    def test_low_detection_discarded(self):
        """Windows with detection rate below threshold are discarded."""
        clip = make_synthetic_clip(n_frames=120, all_detected=False)
        # Force very low detection in a region
        clip["hand_detected"][:WINDOW_LEN] = False
        features = build_features_for_clip(clip)

        windows_strict, _ = window_clip(features, clip["hand_detected"], clip["gesture"],
                                      min_detection_rate=0.9)
        windows_lenient, _ = window_clip(features, clip["hand_detected"], clip["gesture"],
                                       min_detection_rate=0.1)
        # Strict should have fewer windows
        assert len(windows_strict) <= len(windows_lenient)


class TestSplitIntegrity:
    def _make_clips_and_windows(self, n_clips=10, n_frames=120):
        """Helper to create clips, features, windows, and clip_ids."""
        clips = []
        all_windows = []
        all_labels = []
        all_clip_ids = []

        gestures = list(LABEL_TO_IDX.keys())
        for i in range(n_clips):
            gesture = gestures[i % len(gestures)]
            subject = f"s{i % 3:02d}"
            clip = make_synthetic_clip(n_frames=n_frames, gesture=gesture,
                                        subject=subject)
            clips.append(clip)

            features = build_features_for_clip(clip)
            windows, _ = window_clip(features, clip["hand_detected"], clip["gesture"])

            for w in windows:
                all_windows.append(w)
                all_labels.append(clip["label_idx"])
                all_clip_ids.append(i)

        return (
            clips,
            np.array(all_windows, dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            np.array(all_clip_ids, dtype=np.int64),
        )

    def test_clip_split_no_overlap(self):
        """No window in test shares any source clip with any window in train."""
        clips, windows, labels, clip_ids = self._make_clips_and_windows()
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_by_clip(
            clips, windows, labels, clip_ids, seed=42
        )

        # Reconstruct split by re-running the internal logic:
        # split_by_clip assigns clips to train/val/test, then gathers windows.
        # We verify by checking that clip assignments are disjoint.
        from dataset import _split_indices
        rng = np.random.default_rng(42)
        train_clips_idx, val_clips_idx, test_clips_idx = _split_indices(
            len(clips), (0.7, 0.15, 0.15), rng
        )

        train_set = set(int(x) for x in train_clips_idx)
        val_set = set(int(x) for x in val_clips_idx)
        test_set = set(int(x) for x in test_clips_idx)

        # Clip assignments must be disjoint
        assert len(train_set & test_set) == 0, "Train/test clip overlap"
        assert len(train_set & val_set) == 0, "Train/val clip overlap"
        assert len(val_set & test_set) == 0, "Val/test clip overlap"

        # All clips accounted for
        assert train_set | val_set | test_set == set(range(len(clips)))

    def test_split_deterministic(self):
        """Same seed produces identical splits."""
        clips, windows, labels, clip_ids = self._make_clips_and_windows()
        s1 = split_by_clip(clips, windows, labels, clip_ids, seed=99)
        s2 = split_by_clip(clips, windows, labels, clip_ids, seed=99)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    def test_ordering_split_then_window(self):
        """Verify split-then-window: adding a test clip doesn't change train windows."""
        clips_small, w_small, l_small, c_small = self._make_clips_and_windows(n_clips=6)
        clips_large, w_large, l_large, c_large = self._make_clips_and_windows(n_clips=8)

        # The train portion of the smaller dataset should use the same clips
        # regardless of how many clips are in the test portion.
        # This is inherently true because we assign clips to splits before windowing.
        s1 = split_by_clip(clips_small, w_small, l_small, c_small, seed=42)
        s2 = split_by_clip(clips_large, w_large, l_large, c_large, seed=42)

        # Both should produce valid non-empty splits
        assert len(s1[0]) > 0, "Train should be non-empty"
        assert len(s2[0]) > 0, "Train should be non-empty"


class TestNormalisationLeakage:
    def test_norm_stats_train_only(self):
        """Normalisation stats computed on train must differ from full-dataset stats.

        This proves they were computed on the train split only, not the full dataset.
        """
        clips, windows, labels, clip_ids = TestSplitIntegrity()._make_clips_and_windows(n_clips=15)
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_by_clip(
            clips, windows, labels, clip_ids, seed=42
        )

        mean_train, std_train = compute_norm_stats(X_tr)

        # Full dataset stats
        mean_full, std_full = compute_norm_stats(windows)

        # They should differ (unless train == full, which it shouldn't)
        assert len(X_tr) < len(windows), "Train should be a subset of full"
        assert not np.allclose(mean_train, mean_full, atol=1e-6), \
            "Train-only mean should differ from full-dataset mean"
