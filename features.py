"""Phase 2 — Feature Engineering & Invariance.

Pure functions for converting 21 3D hand landmarks into a normalized,
invariant float32 feature vector suitable for classification.

Hard Architectural Boundary:
- NO camera code, NO file I/O, NO OpenCV imports.
- Pure and deterministic functions only.
- Vectorized numpy operations with float32 throughout.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


# Feature vector dimension: 21 landmarks * 3 coordinates (x, y, z)
FEATURE_DIM = 63

# Sentinel vector returned when no hand is present
NO_HAND_SENTINEL = np.zeros(FEATURE_DIM, dtype=np.float32)

# ---------------------------------------------------------------------------
# Landmark Invariance Choices & Rationale
# ---------------------------------------------------------------------------
#
# 1. ORIGIN LANDMARK (Translation Invariance):
#    Index 0: WRIST.
#    Rationale: The wrist is the anatomical root of the hand skeleton. Every
#    finger branch and palm structure originates from the wrist base, making it
#    the most natural and stable origin for relative coordinates.
#
# 2. REFERENCE DISTANCE (Scale Invariance):
#    Distance between Index 0 (WRIST) and Index 9 (MIDDLE_FINGER_MCP).
#    Rationale: The distance between the wrist and the middle finger knuckle
#    represents the rigid length of the palm structure. Unlike wrist-to-fingertip
#    spans, this palm span remains constant regardless of finger flexing or pinching.
#
# 3. HANDEDNESS NORMALIZATION:
#    Invert the x-coordinate of relative landmarks when handedness == "Left".
#    Rationale: Mirroring left hand geometry across the y-axis aligns left and
#    right hands into a single unified 3D feature space.
# ---------------------------------------------------------------------------

ORIGIN_INDEX = 0
SCALE_INDEX = 9
EPSILON = 1e-6


def extract_features(
    landmarks: Optional[Union[np.ndarray, list, tuple]],
    handedness: str = "Right",
    normalize_translation: bool = True,
    normalize_scale: bool = True,
    normalize_handedness: bool = True,
) -> np.ndarray:
    """Extract a 63D float32 normalized feature vector from 21 3D landmarks.

    Args:
        landmarks: Shape (21, 3) array-like of (x, y, z) coordinates, or None.
        handedness: Hand classification ("Right" or "Left"). Defaults to "Right".
        normalize_translation: If True, subtract origin landmark (wrist).
        normalize_scale: If True, scale by wrist-to-middle-MCP distance.
        normalize_handedness: If True, mirror x-coords if handedness == "Left".

    Returns:
        1D float32 numpy array of length 63. Always returns (63,) array.
    """
    if landmarks is None:
        return NO_HAND_SENTINEL.copy()

    # Convert to float32 numpy array
    pts = np.asarray(landmarks, dtype=np.float32)

    # Check for empty or invalid shape
    if pts.size == 0 or pts.shape != (21, 3):
        return NO_HAND_SENTINEL.copy()

    # 1. Translation Invariance (relative to WRIST at index 0)
    if normalize_translation:
        pts = pts - pts[ORIGIN_INDEX]

    # 2. Handedness Normalization (invert x for Left hand)
    if normalize_handedness and str(handedness).strip().lower() == "left":
        # Vectorized x inversion: negate the first column
        pts[:, 0] = -pts[:, 0]

    # 3. Scale Invariance (divide by WRIST-to-MIDDLE_MCP distance)
    if normalize_scale:
        if normalize_translation:
            # Origin is already at pts[0], so distance is norm of pts[SCALE_INDEX]
            ref_dist = float(np.linalg.norm(pts[SCALE_INDEX]))
        else:
            ref_dist = float(np.linalg.norm(pts[SCALE_INDEX] - pts[ORIGIN_INDEX]))

        scale_factor = ref_dist if ref_dist > EPSILON else 1.0
        pts = pts / scale_factor

    # Flatten to 1D float32 array of shape (63,)
    return pts.reshape(-1).astype(np.float32)


def compute_feature_stats(feature_vector: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics (min, max, L2 norm) for a feature vector.

    Args:
        feature_vector: 1D numpy array of shape (63,).

    Returns:
        Dictionary with keys 'min', 'max', 'norm'.
    """
    vec = np.asarray(feature_vector, dtype=np.float32)
    if vec.size == 0:
        return {"min": 0.0, "max": 0.0, "norm": 0.0}

    return {
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
        "norm": float(np.linalg.norm(vec)),
    }
