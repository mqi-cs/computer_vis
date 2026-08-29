"""Phase 2 — Pytest suite for features.py (no camera required)."""

import numpy as np
import pytest

from features import (
    FEATURE_DIM,
    NO_HAND_SENTINEL,
    compute_feature_stats,
    extract_features,
)


@pytest.fixture
def synthetic_hand():
    """Create a realistic 21-point 3D hand landmark array."""
    # Synthetic hand layout centered roughly around wrist at (0.5, 0.5, 0.0)
    base_landmarks = np.array(
        [
            [0.50, 0.70, 0.00],  # 0: Wrist
            [0.45, 0.65, 0.01],  # 1: Thumb CMC
            [0.42, 0.60, 0.02],  # 2: Thumb MCP
            [0.40, 0.55, 0.02],  # 3: Thumb IP
            [0.38, 0.50, 0.03],  # 4: Thumb Tip
            [0.46, 0.52, 0.01],  # 5: Index MCP
            [0.45, 0.45, 0.01],  # 6: Index PIP
            [0.44, 0.38, 0.01],  # 7: Index DIP
            [0.43, 0.32, 0.01],  # 8: Index Tip
            [0.50, 0.50, 0.00],  # 9: Middle MCP
            [0.50, 0.42, 0.00],  # 10: Middle PIP
            [0.50, 0.35, 0.00],  # 11: Middle DIP
            [0.50, 0.28, 0.00],  # 12: Middle Tip
            [0.54, 0.52, -0.01], # 13: Ring MCP
            [0.55, 0.45, -0.01], # 14: Ring PIP
            [0.56, 0.38, -0.01], # 15: Ring DIP
            [0.57, 0.32, -0.01], # 16: Ring Tip
            [0.58, 0.55, -0.02], # 17: Pinky MCP
            [0.60, 0.48, -0.02], # 18: Pinky PIP
            [0.61, 0.42, -0.02], # 19: Pinky DIP
            [0.62, 0.36, -0.02], # 20: Pinky Tip
        ],
        dtype=np.float32,
    )
    return base_landmarks


def test_vector_length_and_dtype(synthetic_hand):
    """Requirement 4: Produce a fixed-length 1D float32 vector (length 63)."""
    feat = extract_features(synthetic_hand)
    print(f"\nExplicit feature vector length: {len(feat)}")
    assert feat.shape == (FEATURE_DIM,), f"Expected shape ({FEATURE_DIM},), got {feat.shape}"
    assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"


def test_translation_invariance(synthetic_hand):
    """Shift all landmarks by constant offset, assert feature vector is unchanged."""
    feat_orig = extract_features(synthetic_hand)

    offset = np.array([12.34, -56.78, 9.10], dtype=np.float32)
    shifted_hand = synthetic_hand + offset

    feat_shifted = extract_features(shifted_hand)
    assert np.allclose(feat_orig, feat_shifted, atol=1e-5)


def test_scale_invariance(synthetic_hand):
    """Multiply landmarks by scale factors (0.2x, 5.0x), assert feature vector is unchanged."""
    feat_orig = extract_features(synthetic_hand)

    for scale in [0.2, 0.5, 2.0, 5.0, 10.0]:
        scaled_hand = synthetic_hand * scale
        feat_scaled = extract_features(scaled_hand)
        assert np.allclose(feat_orig, feat_scaled, atol=1e-5), f"Failed for scale {scale}"


def test_handedness_normalization(synthetic_hand):
    """Mirror right hand (flip x relative to wrist) and mark as Left, assert match."""
    feat_right = extract_features(synthetic_hand, handedness="Right")

    # Mirror hand in 3D relative to wrist x-center
    wrist_x = synthetic_hand[0, 0]
    mirrored_hand = synthetic_hand.copy()
    mirrored_hand[:, 0] = wrist_x - (synthetic_hand[:, 0] - wrist_x)

    feat_left = extract_features(mirrored_hand, handedness="Left")
    assert np.allclose(feat_right, feat_left, atol=1e-5)


def test_determinism(synthetic_hand):
    """Same input twice produces byte-identical output."""
    feat1 = extract_features(synthetic_hand, handedness="Right")
    feat2 = extract_features(synthetic_hand, handedness="Right")

    assert np.array_equal(feat1, feat2)
    assert feat1.tobytes() == feat2.tobytes()


def test_no_hand_sentinel():
    """No-hand case returns documented sentinel vector of shape (63,) zeros."""
    feat_none = extract_features(None)
    feat_empty = extract_features(np.array([]))
    feat_invalid = extract_features(np.zeros((10, 3)))

    assert feat_none.shape == (FEATURE_DIM,)
    assert feat_empty.shape == (FEATURE_DIM,)
    assert feat_invalid.shape == (FEATURE_DIM,)

    assert np.array_equal(feat_none, NO_HAND_SENTINEL)
    assert np.array_equal(feat_empty, NO_HAND_SENTINEL)
    assert np.array_equal(feat_invalid, NO_HAND_SENTINEL)


def test_degenerate_input():
    """All landmarks identical (zero reference distance) must not produce NaN/Inf."""
    zero_hand = np.zeros((21, 3), dtype=np.float32)
    feat_zero = extract_features(zero_hand)

    assert not np.isnan(feat_zero).any()
    assert not np.isinf(feat_zero).any()
    assert feat_zero.shape == (FEATURE_DIM,)

    identical_hand = np.full((21, 3), 42.0, dtype=np.float32)
    feat_identical = extract_features(identical_hand)
    assert not np.isnan(feat_identical).any()
    assert not np.isinf(feat_identical).any()
    assert feat_identical.shape == (FEATURE_DIM,)


def test_ablation_toggles(synthetic_hand):
    """Test feature extraction with individual invariances disabled."""
    offset = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    shifted_hand = synthetic_hand + offset

    # With translation OFF, shifting changes the feature vector
    feat_trans_off_orig = extract_features(synthetic_hand, normalize_translation=False)
    feat_trans_off_shifted = extract_features(shifted_hand, normalize_translation=False)
    assert not np.allclose(feat_trans_off_orig, feat_trans_off_shifted)

    # With scale OFF, scaling changes feature vector norm
    feat_scale_off_1x = extract_features(synthetic_hand, normalize_scale=False)
    feat_scale_off_2x = extract_features(synthetic_hand * 2.0, normalize_scale=False)
    assert not np.allclose(feat_scale_off_1x, feat_scale_off_2x)


def test_feature_stats(synthetic_hand):
    """Test compute_feature_stats output."""
    feat = extract_features(synthetic_hand)
    stats = compute_feature_stats(feat)

    assert "min" in stats and "max" in stats and "norm" in stats
    assert isinstance(stats["min"], float)
    assert isinstance(stats["max"], float)
    assert isinstance(stats["norm"], float)
    assert stats["min"] <= stats["max"]
