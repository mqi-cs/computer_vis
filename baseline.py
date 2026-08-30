"""Phase 4 — Non-Neural Baselines.

Three baselines of increasing sophistication, all using per-window
summary features (mean/std/min/max over time axis → 66×4 = 264D).

1. Majority class — always predicts the most frequent label
2. Logistic Regression (sklearn)
3. Random Forest (sklearn)

Usage:
    uv run baseline.py [--split-dir ./output] [--output-dir ./output]
                       [--split-name by_clip]
"""

import argparse
import pathlib
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend for PNG output
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset import GESTURE_LABELS, IDX_TO_LABEL, apply_norm


# ---------------------------------------------------------------------------
# Window summarisation
# ---------------------------------------------------------------------------

def summarize_windows(X: np.ndarray) -> np.ndarray:
    """Compute per-window summary features: mean/std/min/max over the time axis.

    Args:
        X: (N, window_len, feature_dim) float32

    Returns:
        (N, feature_dim * 4) float32 summary vector per window
    """
    # X has shape (N, T, F)
    return np.concatenate([
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
    ], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Majority class baseline: always predict the most frequent test label (for test-set floor)."""
    # The actual majority floor must be calculated from the test set, not train
    values, counts = np.unique(y_test, return_counts=True)
    majority_label = values[np.argmax(counts)]
    majority_name = IDX_TO_LABEL[majority_label]
    y_pred = np.full_like(y_test, majority_label)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"MAJORITY CLASS BASELINE (on Test Set): always predict '{majority_name}'")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  This is the floor. Any model must beat {acc:.1%}.")
    print(f"{'='*60}")
    return acc


def _check_zero_recall(y_test: np.ndarray, y_pred: np.ndarray):
    """Detect and warn about any classes with zero recall."""
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(GESTURE_LABELS))))
    recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    for i, r in enumerate(recalls):
        # Only warn if the class is actually in the test set
        if cm.sum(axis=1)[i] > 0 and r == 0.0:
            print(f"  ⚠ ZERO RECALL for class: {GESTURE_LABELS[i]}")


def run_logreg_baseline(
    X_train_summary: np.ndarray,
    y_train: np.ndarray,
    X_test_summary: np.ndarray,
    y_test: np.ndarray,
    output_dir: pathlib.Path,
) -> Tuple[float, float]:
    """Logistic Regression baseline on window summary features."""
    clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs",
                             class_weight="balanced")
    clf.fit(X_train_summary, y_train)
    y_pred = clf.predict(X_test_summary)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=GESTURE_LABELS,
        labels=list(range(len(GESTURE_LABELS))),
        zero_division=0,
        output_dict=True
    )
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"\nLogistic Regression (class_weight='balanced')")
    print(f"  Accuracy: {acc:.1%} | Macro-F1: {macro_f1:.1%}")
    print(classification_report(
        y_test, y_pred,
        target_names=GESTURE_LABELS,
        labels=list(range(len(GESTURE_LABELS))),
        zero_division=0,
    ))
    _check_zero_recall(y_test, y_pred)

    save_confusion_matrix(y_test, y_pred, GESTURE_LABELS,
                          output_dir / "cm_logreg.png",
                          f"Logistic Regression (F1: {macro_f1:.1%})")
    return acc, macro_f1


def run_rf_baseline(
    X_train_summary: np.ndarray,
    y_train: np.ndarray,
    X_test_summary: np.ndarray,
    y_test: np.ndarray,
    output_dir: pathlib.Path,
) -> Tuple[float, float]:
    """Random Forest baseline on window summary features."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                 class_weight="balanced")
    clf.fit(X_train_summary, y_train)
    y_pred = clf.predict(X_test_summary)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=GESTURE_LABELS,
        labels=list(range(len(GESTURE_LABELS))),
        zero_division=0,
        output_dict=True
    )
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"\nRandom Forest (class_weight='balanced')")
    print(f"  Accuracy: {acc:.1%} | Macro-F1: {macro_f1:.1%}")
    print(classification_report(
        y_test, y_pred,
        target_names=GESTURE_LABELS,
        labels=list(range(len(GESTURE_LABELS))),
        zero_division=0,
    ))
    _check_zero_recall(y_test, y_pred)

    save_confusion_matrix(y_test, y_pred, GESTURE_LABELS,
                          output_dir / "cm_rf.png",
                          f"Random Forest (F1: {macro_f1:.1%})")
    return acc, macro_f1


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, labels, path, title="Confusion Matrix"):
    """Save a confusion matrix as a PNG image."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    # Cell values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {path}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Baselines")
    parser.add_argument("--split-dir", default="./output")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--split-name", default="by_clip",
                        help="Which split to evaluate (default: by_clip)")
    args = parser.parse_args()

    split_dir = pathlib.Path(args.split_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split
    split_file = split_dir / f"split_{args.split_name}.npz"
    norm_file = split_dir / "norm_stats.npz"

    if not split_file.exists():
        print(f"✗ Split file not found: {split_file}")
        print("  Run dataset.py first.")
        return
    if not norm_file.exists():
        print(f"✗ Norm stats not found: {norm_file}")
        return

    data = np.load(split_file)
    norm = np.load(norm_file)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    mean, std = norm["mean"], norm["std"]

    print(f"Split: {args.split_name}")
    print(f"  Train: {len(y_train)} windows")
    print(f"  Val:   {len(y_val)} windows")
    print(f"  Test:  {len(y_test)} windows")
    
    # Check subjects count (using the fact that split_by_subject logs if < 3)
    # We can just read the JSONs or we can add a simple check
    print("\n--- Split Distributions ---")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        total = len(y)
        if total == 0:
            print(f"  {name}: 0 windows")
            continue
        print(f"  {name} ({total} windows):")
        for idx in range(len(GESTURE_LABELS)):
            count = (y == idx).sum()
            if count > 0:
                print(f"    {GESTURE_LABELS[idx]:<14} {count:>5} ({count/total*100:>5.1f}%)")

    # Normalise
    X_train_n = apply_norm(X_train, mean, std)
    X_test_n = apply_norm(X_test, mean, std)

    # Summarise windows → (N, 264) flat features
    X_train_s = summarize_windows(X_train_n)
    X_test_s = summarize_windows(X_test_n)

    print(f"\n  Summary feature dim: {X_train_s.shape[1]}")

    # --- Baselines ---
    majority_acc = run_majority_baseline(y_train, y_test)
    logreg_acc, logreg_f1 = run_logreg_baseline(X_train_s, y_train, X_test_s, y_test, output_dir)
    rf_acc, rf_f1 = run_rf_baseline(X_train_s, y_train, X_test_s, y_test, output_dir)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY ({args.split_name} split)")
    print(f"{'='*60}")
    print(f"  Majority class floor (Acc): {majority_acc:.1%}")
    print(f"  Logistic Regression:        Acc: {logreg_acc:.1%} | Macro-F1: {logreg_f1:.1%}")
    print(f"  Random Forest:              Acc: {rf_acc:.1%} | Macro-F1: {rf_f1:.1%}")
    print(f"{'='*60}")

    # Run on random_window split too for leakage comparison
    rw_file = split_dir / "split_random_window.npz"
    if rw_file.exists() and args.split_name != "random_window":
        print(f"\n--- Leakage comparison: random_window vs {args.split_name} ---")
        rw_data = np.load(rw_file)
        rw_norm = np.load(norm_file)  # reuse same norm for comparison
        X_rw_train = apply_norm(rw_data["X_train"], mean, std)
        X_rw_test = apply_norm(rw_data["X_test"], mean, std)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
        clf.fit(summarize_windows(X_rw_train), rw_data["y_train"])
        y_rw_pred = clf.predict(summarize_windows(X_rw_test))
        rw_acc = accuracy_score(rw_data["y_test"], y_rw_pred)
        
        report = classification_report(rw_data["y_test"], y_rw_pred, output_dict=True, zero_division=0)
        rw_f1 = report["macro avg"]["f1-score"]

        gap_acc = rw_acc - rf_acc
        gap_f1 = rw_f1 - rf_f1
        
        print(f"  Random Forest on random_window: Acc: {rw_acc:.1%} | Macro-F1: {rw_f1:.1%}")
        print(f"  Random Forest on {args.split_name}:    Acc: {rf_acc:.1%} | Macro-F1: {rf_f1:.1%}")
        print(f"  Leakage Gap:                    Acc: {gap_acc:+.1%} | Macro-F1: {gap_f1:+.1%}")
        if gap_acc > 0.02:
            print("  ✓ Positive gap confirms data leakage in random-window split.")
        else:
            print("  ⚠ Small gap — possibly insufficient data or low leakage conditions.")


if __name__ == "__main__":
    main()
