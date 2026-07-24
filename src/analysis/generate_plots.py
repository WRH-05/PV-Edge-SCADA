"""Publication-ready evaluation plots for EL defect severity models.

Generates five comparative figures between two models (e.g., a baseline
and an SAHL variant):

1. Precision-Recall curve (with Average Precision annotation)
2. ROC curve (with AUC annotation)
3. F1-score vs decision threshold
4. Confusion matrices at an operating threshold
5. Residual (error) distribution histograms

All plots are saved in vector (PDF) and/or raster (PNG) formats, and a
plain-text metrics summary is written alongside the figures.

Typical usage::

    python -m src.analysis.generate_plots \\
        --v1_csv testCsv/v1_report.csv \\
        --v3_csv testCsv/v3_report.csv \\
        --output_dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready model comparison plots: PR/ROC "
            "curves, F1-threshold trends, confusion matrices, and error "
            "distribution charts."
        )
    )
    parser.add_argument(
        "--v1_csv",
        type=str,
        default="testCsv/test_split_report_v1_onnx.csv",
        help="Path to baseline model (V1) test report CSV.",
    )
    parser.add_argument(
        "--v3_csv",
        type=str,
        default="testCsv/test_split_report_v3_1_goldilocks_onnx.csv",
        help="Path to SAHL model (V3.1) test report CSV.",
    )
    parser.add_argument(
        "--v1_name",
        type=str,
        default="V1 (MSE)",
        help="Display name for baseline model in plots and summary.",
    )
    parser.add_argument(
        "--v3_name",
        type=str,
        default="V3.1 (Asymmetric)",
        help="Display name for SAHL model in plots and summary.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save generated figures and summary files.",
    )
    parser.add_argument(
        "--target_threshold",
        type=float,
        default=0.65,
        help="Binary positive label threshold applied on target values.",
    )
    parser.add_argument(
        "--operating_threshold",
        type=float,
        default=0.65,
        help="Operating threshold for confusion-matrix and metric computation.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output figure formats, e.g. png pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Figure DPI for raster exports.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalize_image_path(path_value: str) -> str:
    """Normalise a file path for case-insensitive cross-platform matching.

    Args:
        path_value: Raw path string from a CSV column.

    Returns:
        Lowercased, forward-slash-normalised, stripped string.
    """
    return str(path_value).replace("\\", "/").strip().lower()


def load_and_align_reports(v1_csv: Path, v3_csv: Path) -> pd.DataFrame:
    """Load two test-report CSVs and inner-join them on ``image_path``.

    The CSVs must each contain columns ``image_path``, ``target``,
    ``prediction``, ``abs_error``, and ``squared_error``.  Targets are
    validated for consistency across the two reports.

    Args:
        v1_csv: Path to the baseline model's test report.
        v3_csv: Path to the SAHL model's test report.

    Returns:
        A merged DataFrame with columns suffixed ``_v1`` and ``_v3``.

    Raises:
        ValueError: If required columns are missing, no overlapping images
            are found, or target values differ between reports.
    """
    required_columns = {
        "image_path", "target", "prediction", "abs_error", "squared_error"
    }

    v1_df = pd.read_csv(v1_csv)
    v3_df = pd.read_csv(v3_csv)

    for label, df, csv_path in [
        ("V1", v1_df, v1_csv),
        ("V3.1", v3_df, v3_csv),
    ]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {label} CSV ({csv_path}): {missing}"
            )

    v1_df = v1_df.copy()
    v3_df = v3_df.copy()

    v1_df["image_key"] = v1_df["image_path"].astype(str).map(_normalize_image_path)
    v3_df["image_key"] = v3_df["image_path"].astype(str).map(_normalize_image_path)

    merged = v1_df.merge(
        v3_df,
        on="image_key",
        suffixes=("_v1", "_v3"),
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No overlapping image paths found between V1 and V3.1 reports."
        )

    # Verify that both reports were generated from the same held-out split
    target_mismatch = np.abs(merged["target_v1"] - merged["target_v3"]) > 1e-6
    if target_mismatch.any():
        mismatch_count = int(target_mismatch.sum())
        raise ValueError(
            f"Target mismatch found for {mismatch_count} aligned rows. "
            "Ensure both reports are generated from the same test split and seed."
        )

    return merged


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_model_metrics(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray,
    y_true_continuous: np.ndarray,
    operating_threshold: float,
) -> Dict[str, object]:
    """Compute a comprehensive suite of model evaluation metrics.

    Args:
        y_true_binary: Binary ground-truth labels (0 or 1).
        y_scores: Continuous prediction scores in ``[0, 1]``.
        y_true_continuous: Continuous ground-truth values.
        operating_threshold: Decision threshold for binarisation.

    Returns:
        Dictionary with keys: ``precision_curve``, ``recall_curve``,
        ``fpr_curve``, ``tpr_curve``, ``ap``, ``roc_auc``,
        ``precision_at_op``, ``recall_at_op``, ``f1_at_op``, ``mae``,
        ``mse``, ``residuals``, ``confusion_matrix``.
    """
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)

    y_pred_binary = (y_scores >= operating_threshold).astype(int)
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    return {
        "precision_curve": precision,
        "recall_curve": recall,
        "fpr_curve": fpr,
        "tpr_curve": tpr,
        "ap": float(average_precision_score(y_true_binary, y_scores)),
        "roc_auc": float(roc_auc_score(y_true_binary, y_scores)),
        "precision_at_op": float(
            precision_score(y_true_binary, y_pred_binary, zero_division=0)
        ),
        "recall_at_op": float(
            recall_score(y_true_binary, y_pred_binary, zero_division=0)
        ),
        "f1_at_op": float(
            f1_score(y_true_binary, y_pred_binary, zero_division=0)
        ),
        "mae": float(np.mean(np.abs(y_scores - y_true_continuous))),
        "mse": float(np.mean((y_scores - y_true_continuous) ** 2)),
        "residuals": y_scores - y_true_continuous,
        "confusion_matrix": cm,
    }


def compute_f1_threshold_curve(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall, and F1 across a sweep of thresholds.

    Args:
        y_true_binary: Binary ground-truth labels.
        y_scores: Continuous prediction scores.
        thresholds: Array of threshold values to evaluate.

    Returns:
        ``(precision_values, recall_values, f1_values, thresholds)`` as
        float64 numpy arrays.
    """
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []

    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        precision_values.append(
            float(precision_score(y_true_binary, y_pred, zero_division=0))
        )
        recall_values.append(
            float(recall_score(y_true_binary, y_pred, zero_division=0))
        )
        f1_values.append(
            float(f1_score(y_true_binary, y_pred, zero_division=0))
        )

    return (
        np.array(precision_values, dtype=np.float64),
        np.array(recall_values, dtype=np.float64),
        np.array(f1_values, dtype=np.float64),
        thresholds,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    base_path: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Save a matplotlib figure in each requested format.

    Args:
        fig: The matplotlib figure to save.
        base_path: Base filename without extension.
        formats: Iterable of extensions (e.g. ``["png", "pdf"]``).
        dpi: Raster resolution.
    """
    for fmt in formats:
        fmt_lower = fmt.lower()
        out_path = base_path.with_suffix(f".{fmt_lower}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------


def plot_pr_curve(
    v1: Dict[str, object],
    v3: Dict[str, object],
    v1_name: str,
    v3_name: str,
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Plot Precision-Recall curves for two models with AP annotation.

    A high-recall region (recall ≥ 0.8) is highlighted to support the
    results-section narrative about safety-critical detection.

    Args:
        v1: Metrics dict for the baseline model.
        v3: Metrics dict for the SAHL model.
        v1_name: Display name for baseline.
        v3_name: Display name for SAHL.
        out_dir: Output directory.
        formats: Output file formats.
        dpi: Raster DPI.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        v1["recall_curve"], v1["precision_curve"],
        label=f"{v1_name} (AP={v1['ap']:.4f})", linewidth=2,
    )
    ax.plot(
        v3["recall_curve"], v3["precision_curve"],
        label=f"{v3_name} (AP={v3['ap']:.4f})", linewidth=2,
    )
    ax.set_title(f"Precision-Recall Curve: {v1_name} vs {v3_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    # Highlight high-recall operating region
    ax.axvspan(0.8, 1.0, color="orange", alpha=0.08)
    ax.text(0.805, 0.08, "High-recall region", fontsize=9)

    save_figure(fig, out_dir / "pr_curve_v1_vs_v3_1", formats=formats, dpi=dpi)
    plt.close(fig)


def plot_roc_curve(
    v1: Dict[str, object],
    v3: Dict[str, object],
    v1_name: str,
    v3_name: str,
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Plot ROC curves for two models with AUC annotation.

    Args:
        v1: Metrics dict for the baseline model.
        v3: Metrics dict for the SAHL model.
        v1_name: Display name for baseline.
        v3_name: Display name for SAHL.
        out_dir: Output directory.
        formats: Output file formats.
        dpi: Raster DPI.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        v1["fpr_curve"], v1["tpr_curve"],
        label=f"{v1_name} (AUC={v1['roc_auc']:.4f})", linewidth=2,
    )
    ax.plot(
        v3["fpr_curve"], v3["tpr_curve"],
        label=f"{v3_name} (AUC={v3['roc_auc']:.4f})", linewidth=2,
    )
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", linewidth=1, color="gray", label="Random",
    )
    ax.set_title(f"ROC Curve: {v1_name} vs {v3_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    save_figure(
        fig, out_dir / "roc_curve_v1_vs_v3_1", formats=formats, dpi=dpi
    )
    plt.close(fig)


def plot_f1_threshold(
    f1_curve_v1: np.ndarray,
    f1_curve_v3: np.ndarray,
    thresholds: np.ndarray,
    operating_threshold: float,
    v1_name: str,
    v3_name: str,
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Plot F1 score as a function of the decision threshold.

    Best-F1 points are marked with scatter dots, and the operating
    threshold is drawn as a vertical reference line.

    Args:
        f1_curve_v1: F1 values for baseline at each threshold.
        f1_curve_v3: F1 values for SAHL model.
        thresholds: Array of threshold values.
        operating_threshold: The chosen operating threshold.
        v1_name: Display name for baseline.
        v3_name: Display name for SAHL.
        out_dir: Output directory.
        formats: Output file formats.
        dpi: Raster DPI.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, f1_curve_v1, label=f"{v1_name} F1", linewidth=2)
    ax.plot(thresholds, f1_curve_v3, label=f"{v3_name} F1", linewidth=2)

    best_idx_v1 = int(np.argmax(f1_curve_v1))
    best_idx_v3 = int(np.argmax(f1_curve_v3))
    ax.scatter(thresholds[best_idx_v1], f1_curve_v1[best_idx_v1], s=30)
    ax.scatter(thresholds[best_idx_v3], f1_curve_v3[best_idx_v3], s=30)

    ax.axvline(
        operating_threshold,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=f"Operating threshold={operating_threshold:.2f}",
    )
    ax.set_title("F1 Score vs Prediction Threshold")
    ax.set_xlabel("Prediction Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    save_figure(fig, out_dir / "f1_vs_threshold", formats=formats, dpi=dpi)
    plt.close(fig)


def _annotate_confusion_matrix(
    ax: plt.Axes, cm: np.ndarray, title: str
) -> None:
    """Draw a single annotated confusion-matrix heatmap.

    Each cell displays the raw count and the row-normalised percentage.

    Args:
        ax: Matplotlib Axes to draw on.
        cm: 2×2 confusion matrix.
        title: Subplot title.
    """
    cm_float = cm.astype(np.float64)
    total = cm_float.sum()
    normalized = cm_float / total if total > 0 else cm_float

    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Non-defect", "Defect"])
    ax.set_yticks([0, 1], labels=["Non-defect", "Defect"])

    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            pct = 100.0 * normalized[i, j] if total > 0 else 0.0
            text_color = "white" if normalized[i, j] > 0.35 else "black"
            ax.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color, fontsize=10,
            )


def plot_confusion_matrices(
    cm_v1: np.ndarray,
    cm_v3: np.ndarray,
    operating_threshold: float,
    v1_name: str,
    v3_name: str,
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Plot side-by-side confusion matrices for two models.

    Args:
        cm_v1: 2×2 confusion matrix for baseline.
        cm_v3: 2×2 confusion matrix for SAHL model.
        operating_threshold: Threshold used for binarisation.
        v1_name: Display name for baseline.
        v3_name: Display name for SAHL.
        out_dir: Output directory.
        formats: Output file formats.
        dpi: Raster DPI.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _annotate_confusion_matrix(
        axes[0], cm_v1, f"{v1_name} @ threshold {operating_threshold:.2f}"
    )
    _annotate_confusion_matrix(
        axes[1], cm_v3, f"{v3_name} @ threshold {operating_threshold:.2f}"
    )
    fig.suptitle("Confusion Matrix Comparison")

    save_figure(
        fig,
        out_dir / "confusion_matrices_at_operating_threshold",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)


def plot_error_distribution(
    residuals_v1: np.ndarray,
    residuals_v3: np.ndarray,
    v1_name: str,
    v3_name: str,
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Plot overlaid histograms of prediction residuals.

    Residuals are computed as ``prediction - target``, so a positive
    residual indicates over-prediction.

    Args:
        residuals_v1: Residual array for baseline model.
        residuals_v3: Residual array for SAHL model.
        v1_name: Display name for baseline.
        v3_name: Display name for SAHL.
        out_dir: Output directory.
        formats: Output file formats.
        dpi: Raster DPI.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(-1.0, 1.0, 60)
    ax.hist(
        residuals_v1, bins=bins,
        alpha=0.45, density=True,
        label=f"{v1_name} residuals", color="#1f77b4",
    )
    ax.hist(
        residuals_v3, bins=bins,
        alpha=0.45, density=True,
        label=f"{v3_name} residuals", color="#d62728",
    )
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_title("Residual Distribution (prediction - target)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    save_figure(
        fig,
        out_dir / "error_distribution_comparison",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_metrics_summary(
    out_dir: Path,
    v1_metrics: Dict[str, object],
    v3_metrics: Dict[str, object],
    v1_name: str,
    v3_name: str,
    target_threshold: float,
    operating_threshold: float,
) -> None:
    """Write a human-readable metrics comparison to a text file.

    Args:
        out_dir: Output directory.
        v1_metrics: Metrics dict for the baseline model.
        v3_metrics: Metrics dict for the SAHL model.
        v1_name: Display name for baseline.
        v3_name: Display name for SAHL.
        target_threshold: Threshold used to binarise targets.
        operating_threshold: Threshold used to binarise predictions.
    """
    lines = [
        f"{v1_name} vs {v3_name} Comparative Summary",
        "=" * 38,
        f"Target binarization threshold: {target_threshold:.4f}",
        f"Operating decision threshold: {operating_threshold:.4f}",
        "",
        "Primary discrimination metrics",
        f"- {v1_name} Average Precision (AP): {v1_metrics['ap']:.6f}",
        f"- {v3_name} Average Precision (AP): {v3_metrics['ap']:.6f}",
        f"- {v1_name} ROC-AUC: {v1_metrics['roc_auc']:.6f}",
        f"- {v3_name} ROC-AUC: {v3_metrics['roc_auc']:.6f}",
        "",
        f"Operating-point metrics @ threshold {operating_threshold:.2f}",
        f"- {v1_name} Precision: {v1_metrics['precision_at_op']:.6f}",
        f"- {v3_name} Precision: {v3_metrics['precision_at_op']:.6f}",
        f"- {v1_name} Recall: {v1_metrics['recall_at_op']:.6f}",
        f"- {v3_name} Recall: {v3_metrics['recall_at_op']:.6f}",
        f"- {v1_name} F1: {v1_metrics['f1_at_op']:.6f}",
        f"- {v3_name} F1: {v3_metrics['f1_at_op']:.6f}",
        "",
        "Regression fidelity metrics",
        f"- {v1_name} MAE: {v1_metrics['mae']:.6f}",
        f"- {v3_name} MAE: {v3_metrics['mae']:.6f}",
        f"- {v1_name} MSE: {v1_metrics['mse']:.6f}",
        f"- {v3_name} MSE: {v3_metrics['mse']:.6f}",
    ]

    summary_path = out_dir / "metrics_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved summary: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate all plot generation and metric summarisation."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_align_reports(Path(args.v1_csv), Path(args.v3_csv))

    y_true_continuous = merged["target_v1"].to_numpy(dtype=np.float64)
    y_true_binary = (y_true_continuous >= args.target_threshold).astype(int)

    y_score_v1 = merged["prediction_v1"].to_numpy(dtype=np.float64)
    y_score_v3 = merged["prediction_v3"].to_numpy(dtype=np.float64)

    positive_count = int(y_true_binary.sum())
    negative_count = int((1 - y_true_binary).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Binary labels collapse into a single class. "
            "Adjust --target_threshold to get both classes."
        )

    print(f"Aligned samples: {len(merged)}")
    print(f"Binary positives: {positive_count}, negatives: {negative_count}")

    v1_metrics = compute_model_metrics(
        y_true_binary=y_true_binary,
        y_scores=y_score_v1,
        y_true_continuous=y_true_continuous,
        operating_threshold=args.operating_threshold,
    )
    v3_metrics = compute_model_metrics(
        y_true_binary=y_true_binary,
        y_scores=y_score_v3,
        y_true_continuous=y_true_continuous,
        operating_threshold=args.operating_threshold,
    )

    thresholds = np.linspace(0.0, 1.0, 201)
    _, _, f1_curve_v1, f1_thresholds = compute_f1_threshold_curve(
        y_true_binary, y_score_v1, thresholds
    )
    _, _, f1_curve_v3, _ = compute_f1_threshold_curve(
        y_true_binary, y_score_v3, thresholds
    )

    plot_pr_curve(
        v1_metrics, v3_metrics,
        v1_name=args.v1_name, v3_name=args.v3_name,
        out_dir=output_dir, formats=args.formats, dpi=args.dpi,
    )
    plot_roc_curve(
        v1_metrics, v3_metrics,
        v1_name=args.v1_name, v3_name=args.v3_name,
        out_dir=output_dir, formats=args.formats, dpi=args.dpi,
    )
    plot_f1_threshold(
        f1_curve_v1=f1_curve_v1, f1_curve_v3=f1_curve_v3,
        thresholds=f1_thresholds,
        operating_threshold=args.operating_threshold,
        v1_name=args.v1_name, v3_name=args.v3_name,
        out_dir=output_dir, formats=args.formats, dpi=args.dpi,
    )
    plot_confusion_matrices(
        cm_v1=v1_metrics["confusion_matrix"],
        cm_v3=v3_metrics["confusion_matrix"],
        operating_threshold=args.operating_threshold,
        v1_name=args.v1_name, v3_name=args.v3_name,
        out_dir=output_dir, formats=args.formats, dpi=args.dpi,
    )
    plot_error_distribution(
        residuals_v1=v1_metrics["residuals"],
        residuals_v3=v3_metrics["residuals"],
        v1_name=args.v1_name, v3_name=args.v3_name,
        out_dir=output_dir, formats=args.formats, dpi=args.dpi,
    )

    write_metrics_summary(
        out_dir=output_dir,
        v1_metrics=v1_metrics,
        v3_metrics=v3_metrics,
        v1_name=args.v1_name,
        v3_name=args.v3_name,
        target_threshold=args.target_threshold,
        operating_threshold=args.operating_threshold,
    )

    print(f"Saved figures and summary to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
