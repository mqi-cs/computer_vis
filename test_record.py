"""Phase 3 — Pytest suite for record.py (no camera required)."""

import json
import pathlib
import pytest
import numpy as np

from record import (
    clip_filename,
    next_clip_index,
    save_clip_atomic,
    print_dataset_summary,
)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Provides a temporary directory for test outputs."""
    return tmp_path / "data"


def test_atomic_write_and_schema(temp_data_dir):
    """Test atomic saving of .npz and .json and verify array shapes/dtypes."""
    basename = clip_filename("PINCH", "s01", "sess1", 0)

    # Create dummy buffers for 5 frames
    n_frames = 5
    arrays = {
        "landmarks": np.random.rand(n_frames, 21, 3).astype(np.float32),
        "world_landmarks": np.random.rand(n_frames, 21, 3).astype(np.float32),
        "timestamps": np.arange(n_frames, dtype=np.int64),
        "handedness": np.array(["Right"] * n_frames, dtype="<U10"),
        "hand_detected": np.ones(n_frames, dtype=bool),
    }

    metadata = {
        "gesture": "PINCH",
        "subject": "s01",
        "session": "sess1",
        "hand_used": "right",
        "lighting": "normal",
        "camera_distance": "normal",
        "duration_sec": 0.16,
        "frame_count": n_frames,
        "hand_detected_count": n_frames,
        "detection_rate": 1.0,
        "tool_version": "v0.3.0",
        "utc_timestamp": "2026-08-29T12:00:00Z",
        "filename_npz": f"{basename}.npz",
    }

    npz_path, json_path = save_clip_atomic(temp_data_dir, basename, arrays, metadata)

    assert npz_path.exists()
    assert json_path.exists()

    # Verify no .tmp files linger
    assert len(list(temp_data_dir.glob("*.tmp.npz"))) == 0
    assert len(list(temp_data_dir.glob("*.tmp.json"))) == 0

    # Load and verify NPZ
    with np.load(npz_path) as data:
        assert set(data.files) == {"landmarks", "world_landmarks", "timestamps", "handedness", "hand_detected"}
        assert data["landmarks"].shape == (n_frames, 21, 3)
        assert data["landmarks"].dtype == np.float32
        assert data["world_landmarks"].shape == (n_frames, 21, 3)
        assert data["world_landmarks"].dtype == np.float32
        assert data["timestamps"].shape == (n_frames,)
        assert data["timestamps"].dtype == np.int64
        assert data["handedness"].shape == (n_frames,)
        assert data["handedness"].dtype.kind == "U"  # Unicode string
        assert data["hand_detected"].shape == (n_frames,)
        assert data["hand_detected"].dtype == bool

    # Load and verify JSON
    with open(json_path) as f:
        meta_loaded = json.load(f)
        assert meta_loaded == metadata


def test_filename_indexing(temp_data_dir):
    """Test index auto-increment prevents overwriting."""
    temp_data_dir.mkdir(parents=True)

    # Should start at 0
    idx1 = next_clip_index(temp_data_dir, "SWIPE_LEFT", "s01", "sess1")
    assert idx1 == 0

    # Create dummy file at index 0
    (temp_data_dir / "SWIPE_LEFT__s01__sess1__0000.npz").touch()

    # Next index should be 1
    idx2 = next_clip_index(temp_data_dir, "SWIPE_LEFT", "s01", "sess1")
    assert idx2 == 1

    # Create dummy file at index 5
    (temp_data_dir / "SWIPE_LEFT__s01__sess1__0005.npz").touch()

    # Next index should be 6
    idx3 = next_clip_index(temp_data_dir, "SWIPE_LEFT", "s01", "sess1")
    assert idx3 == 6

    # Different gesture should start at 0
    idx_other = next_clip_index(temp_data_dir, "ARM", "s01", "sess1")
    assert idx_other == 0


def test_dataset_summary(temp_data_dir, capsys):
    """Test dataset summary runs without error on dummy data."""
    # Write a dummy metadata file
    temp_data_dir.mkdir()
    meta = {
        "gesture": "PINCH",
        "subject": "s01",
        "frame_count": 100,
        "hand_detected_count": 95,
        "duration_sec": 3.3,
    }
    with open(temp_data_dir / "PINCH__s01__sess1__0000.json", "w") as f:
        json.dump(meta, f)

    # Run summary and capture output
    print_dataset_summary(temp_data_dir)
    captured = capsys.readouterr()

    # Check output contains expected values
    assert "PINCH" in captured.out
    assert "100" in captured.out  # frames
    assert "95.0%" in captured.out # detection rate
    assert "s01" in captured.out  # subject
