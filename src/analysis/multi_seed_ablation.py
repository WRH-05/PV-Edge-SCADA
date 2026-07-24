"""Multi-seed SAHL weight ablation study with statistical significance testing.

This module orchestrates the train → export → report pipeline for multiple
SAHL weight configurations across multiple random seeds, then aggregates
the resulting test CSV reports into publication-ready Markdown tables with
per-weight Mean ± Std metrics and paired significance tests.

Two study modes are available:

* **ablation** — 4 SAHL weight multipliers (1.0×, 1.5×, 2.5×, 5.0×) at a
  single fixed seed (42).
* **variance** — 3 seeds (42, 123, 2026) for each of two model variants
  (MSE baseline and SAHL at a configurable weight), producing tables
  across seeds.

Typical usage::

    python -m src.analysis.multi_seed_ablation --mode ablation
    python -m src.analysis.multi_seed_ablation --mode variance
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS: Sequence[float] = (1.0, 1.5, 2.5, 5.0)
"""SAHL weight multipliers evaluated in the ablation sweep."""

SEEDS: Sequence[int] = (42, 123, 2026)
"""Random seeds used for multi-seed variance assessment."""

GENERAL_THRESHOLD: float = 0.65
"""Default binarisation threshold for F1 / precision / recall."""

CRITICAL_PREDICTION_THRESHOLD: float = 0.6
"""Prediction threshold for critical-recall computation."""

CRITICAL_TARGET_THRESHOLD: float = 0.8
"""Target threshold defining a safety-critical sample."""

# Paths to scripts in the src/training/ package (relative to project root).
_TRAIN_SCRIPT = "src/training/train.py"
_EXPORT_SCRIPT = "src/training/export_onnx.py"
_REPORT_SCRIPT = "evaluate_test_split_report.py"


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Immutable configuration for a single training experiment.

    Attributes:
        seed: Random seed.
        loss_type: Loss identifier (``"mse"``, ``"weighted_l1"``, …).
        loss_weight_multiplier: Critical-sample weight for asymmetric losses.
        loss_weight_threshold: Target threshold for asymmetric weighting.
        critical_recall_threshold: Prediction threshold for critical recall.
        critical_target_threshold: Target threshold for critical samples.
        precision_floor: Minimum precision before checkpoint saves are
            permitted.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        csv_path: Path to the labels CSV.
        data_root: Root directory for resolving image paths.
        image_size: Input image dimension (square).
        device: Torch device (``"cuda"`` or ``"cpu"``).
    """

    seed: int
    loss_type: str
    loss_weight_multiplier: float
    loss_weight_threshold: float = 0.70
    critical_recall_threshold: float = 0.6
    critical_target_threshold: float = 0.8
    precision_floor: float = 0.70
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    csv_path: str = "labels.csv"
    data_root: str = "."
    image_size: int = 224
    device: str = "cuda"

    def get_checkpoint_name(self) -> str:
        """Generate a unique checkpoint filename for this configuration."""
        return (
            f"best_model_seed{self.seed}_"
            f"{self.loss_type}_w{self.loss_weight_multiplier:.1f}.pth"
        )

    def get_onnx_name(self) -> str:
        """Generate a unique ONNX filename for this configuration."""
        return (
            f"best_model_seed{self.seed}_"
            f"{self.loss_type}_w{self.loss_weight_multiplier:.1f}.onnx"
        )

    def get_report_name(self) -> str:
        """Generate a unique report CSV filename for this configuration."""
        return (
            f"test_split_report_seed{self.seed}_"
            f"{self.loss_type}_w{self.loss_weight_multiplier:.1f}.csv"
        )


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def _invoke_python_step(
    step_name: str,
    script_path: str,
    extra_args: List[str],
) -> None:
    """Run a Python script as a subprocess, checking for errors.

    Uses ``sys.executable`` to ensure the same interpreter is used across
    all steps, regardless of virtual environment setup.

    Args:
        step_name: Human-readable label for the step.
        script_path: Path to the Python script (relative to CWD).
        extra_args: Additional CLI arguments for the script.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero code.
    """
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n{'='*70}")
    print(f"  {step_name}")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with exit code {result.returncode}"
        )


def run_single_experiment(config: ExperimentConfig) -> None:
    """Execute train → export → report for a single configuration.

    Supports resume: if the final report CSV already exists the entire
    experiment is skipped; individual steps are also skipped when their
    output artifacts are present on disk.

    Args:
        config: The experiment configuration to run.
    """
    checkpoint_path = Path("models/pth") / config.get_checkpoint_name()
    onnx_path = Path("models/onnx") / config.get_onnx_name()
    report_path = Path("testCsv") / config.get_report_name()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\nRunning experiment: {config.loss_type} @ "
        f"{config.loss_weight_multiplier}x, seed={config.seed}"
    )
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  ONNX: {onnx_path}")
    print(f"  Report: {report_path}")

    if report_path.exists():
        print("  Resume mode: report already exists, skipping this experiment.")
        return

    # Step 1: Train
    if not checkpoint_path.exists():
        train_args = [
            "--csv_path", config.csv_path,
            "--data_root", config.data_root,
            "--epochs", str(config.epochs),
            "--batch_size", str(config.batch_size),
            "--learning_rate", str(config.learning_rate),
            "--seed", str(config.seed),
            "--checkpoint_path", str(checkpoint_path),
            "--loss_type", config.loss_type,
            "--loss_weight_threshold", str(config.loss_weight_threshold),
            "--loss_weight_multiplier", str(config.loss_weight_multiplier),
            "--critical_recall_threshold",
            str(config.critical_recall_threshold),
            "--critical_target_threshold",
            str(config.critical_target_threshold),
            "--precision_floor", str(config.precision_floor),
            "--image_size", str(config.image_size),
            "--device", config.device,
        ]
        _invoke_python_step("Train", _TRAIN_SCRIPT, train_args)
    else:
        print("  Resume mode: checkpoint already exists, skipping train step.")

    # Step 2: Export ONNX
    if not onnx_path.exists():
        export_args = [
            "--checkpoint", str(checkpoint_path),
            "--onnx_output", str(onnx_path),
            "--image_size", str(config.image_size),
        ]
        _invoke_python_step("Export ONNX", _EXPORT_SCRIPT, export_args)
    else:
        print("  Resume mode: ONNX already exists, skipping export step.")

    # Step 3: Generate test report
    if not report_path.exists():
        report_args = [
            "--csv_path", config.csv_path,
            "--data_root", config.data_root,
            "--seed", str(config.seed),
            "--image_size", str(config.image_size),
            "--onnx_model", str(onnx_path),
            "--output_csv", str(report_path),
        ]
        _invoke_python_step("Generate Test Report", _REPORT_SCRIPT, report_args)


# ---------------------------------------------------------------------------
# Metric functions (inlined from aggregate_results.py)
# ---------------------------------------------------------------------------


def load_test_report(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a test report CSV and return predictions and targets.

    Args:
        csv_path: Path to a ``test_split_report_*.csv`` file.

    Returns:
        ``(predictions, targets)`` as float64 numpy arrays.
    """
    df = pd.read_csv(csv_path)
    predictions = df["prediction"].to_numpy(dtype=np.float64)
    targets = df["target"].to_numpy(dtype=np.float64)
    return predictions, targets


def compute_metrics_at_threshold(
    predictions: np.ndarray,
    targets: np.ndarray,
    prediction_threshold: float = 0.65,
    target_threshold: float = 0.65,
) -> Dict[str, float]:
    """Compute F1, precision, and recall at a fixed decision threshold.

    Args:
        predictions: Continuous predictions in ``[0, 1]``.
        targets: Continuous targets in ``[0, 1]``.
        prediction_threshold: Threshold to binarise predictions.
        target_threshold: Threshold to binarise targets.

    Returns:
        Dictionary with keys ``"f1"``, ``"precision"``, ``"recall"``.
    """
    y_true_binary = (targets >= target_threshold).astype(int)
    y_pred_binary = (predictions >= prediction_threshold).astype(int)

    return {
        "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        "precision": float(
            precision_score(y_true_binary, y_pred_binary, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_binary, y_pred_binary, zero_division=0)
        ),
    }


def compute_critical_recall(
    predictions: np.ndarray,
    targets: np.ndarray,
    prediction_threshold: float = 0.6,
    target_threshold: float = 0.8,
) -> float:
    """Compute recall restricted to safety-critical (high-severity) samples.

    Args:
        predictions: Continuous predictions.
        targets: Continuous targets.
        prediction_threshold: Threshold to count a prediction as positive.
        target_threshold: Threshold for a sample to be considered critical.

    Returns:
        Critical recall value in ``[0.0, 1.0]``.
    """
    critical_mask = targets >= target_threshold
    critical_count = int(critical_mask.sum())
    if critical_count == 0:
        return 0.0

    true_positives = int(
        (predictions[critical_mask] >= prediction_threshold).sum()
    )
    return float(true_positives / critical_count)


def compute_mae(
    predictions: np.ndarray, targets: np.ndarray
) -> float:
    """Compute Mean Absolute Error.

    Args:
        predictions: Continuous predictions.
        targets: Continuous targets.

    Returns:
        MAE as a float.
    """
    return float(np.mean(np.abs(predictions - targets)))


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _report_path_for(seed: int, weight: float) -> Path:
    """Return the expected test-report path for a given seed and weight.

    Args:
        seed: Random seed.
        weight: SAHL weight multiplier.

    Returns:
        Path relative to the project root.
    """
    return (
        Path("testCsv")
        / f"test_split_report_seed{seed}_weighted_l1_w{weight:.1f}.csv"
    )


def format_mean_std(values: Sequence[float], decimals: int) -> str:
    """Format a sequence of values as ``"mean ± std"``.

    Args:
        values: Numeric values.
        decimals: Number of decimal places.

    Returns:
        A string like ``"0.852 ± 0.003"``.
    """
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def summarize_reports(
    weights: Sequence[float],
    seeds: Sequence[int],
) -> Tuple[List[Dict[str, str]], Dict[float, Dict[str, List[float]]]]:
    """Aggregate test reports across weights × seeds into summary rows.

    Args:
        weights: SAHL weight multipliers to evaluate.
        seeds: Random seeds to evaluate.

    Returns:
        ``(summary_rows, raw_metrics)`` where *summary_rows* is a list of
        dicts ready for Markdown rendering and *raw_metrics* is a nested
        dict mapping weight → metric_name → list of per-seed values.

    Raises:
        FileNotFoundError: If any expected test report is missing.
    """
    summary_rows: List[Dict[str, str]] = []
    raw_metrics: Dict[float, Dict[str, List[float]]] = {
        weight: {
            "f1": [], "precision": [], "recall": [],
            "critical_recall": [], "mae": [],
        }
        for weight in weights
    }

    for weight in weights:
        for seed in seeds:
            report_path = _report_path_for(seed, weight)
            if not report_path.exists():
                raise FileNotFoundError(
                    f"Expected test report not found: {report_path}. "
                    "Run the training/export pipeline first."
                )

            predictions, targets = load_test_report(report_path)
            threshold_metrics = compute_metrics_at_threshold(
                predictions, targets,
                prediction_threshold=GENERAL_THRESHOLD,
                target_threshold=GENERAL_THRESHOLD,
            )

            raw_metrics[weight]["f1"].append(threshold_metrics["f1"])
            raw_metrics[weight]["precision"].append(
                threshold_metrics["precision"]
            )
            raw_metrics[weight]["recall"].append(threshold_metrics["recall"])
            raw_metrics[weight]["critical_recall"].append(
                compute_critical_recall(
                    predictions, targets,
                    prediction_threshold=CRITICAL_PREDICTION_THRESHOLD,
                    target_threshold=CRITICAL_TARGET_THRESHOLD,
                )
            )
            raw_metrics[weight]["mae"].append(
                compute_mae(predictions, targets)
            )

        summary_rows.append(
            {
                "weight": f"{weight:.1f}x",
                "f1": format_mean_std(raw_metrics[weight]["f1"], 3),
                "precision": format_mean_std(
                    raw_metrics[weight]["precision"], 3
                ),
                "recall": format_mean_std(
                    raw_metrics[weight]["recall"], 3
                ),
                "critical_recall": format_mean_std(
                    raw_metrics[weight]["critical_recall"], 3
                ),
                "mae": format_mean_std(raw_metrics[weight]["mae"], 4),
            }
        )

    return summary_rows, raw_metrics


def paired_p_value(
    sample_a: Sequence[float], sample_b: Sequence[float]
) -> Tuple[str, float]:
    """Compute a paired significance test between two matched samples.

    Uses a paired t-test (``scipy.stats.ttest_rel``) first.  If the
    p-value is NaN (e.g., due to zero variance), falls back to the
    Wilcoxon signed-rank test.

    Args:
        sample_a: First set of measurements.
        sample_b: Second set of measurements (same length).

    Returns:
        ``(method_name, p_value)`` tuple.
    """
    ttest = stats.ttest_rel(sample_a, sample_b)
    p_value = float(ttest.pvalue)
    method = "ttest_rel"

    if math.isnan(p_value):
        wilcoxon = stats.wilcoxon(
            sample_a, sample_b, zero_method="wilcox", alternative="two-sided"
        )
        p_value = float(wilcoxon.pvalue)
        method = "wilcoxon"

    return method, p_value


def write_markdown_summary(
    output_path: Path,
    summary_rows: Sequence[Dict[str, str]],
) -> None:
    """Write ablation summary rows as a Markdown table.

    Args:
        output_path: Destination ``.md`` file path.
        summary_rows: List of per-weight dicts with keys ``"weight"``,
            ``"f1"``, ``"precision"``, ``"recall"``, ``"critical_recall"``,
            ``"mae"``.
    """
    lines = [
        "# Multi-Seed SAHL Weight Ablation Summary",
        "",
        "Metrics are reported as Mean ± Std across seeds "
        f"{tuple(SEEDS)}.",
        (
            f"General metrics use threshold {GENERAL_THRESHOLD:.2f}; "
            f"critical recall uses pred >= {CRITICAL_PREDICTION_THRESHOLD:.1f} "
            f"and target >= {CRITICAL_TARGET_THRESHOLD:.1f}."
        ),
        "",
        "| Weight | F1-Score | Precision | Recall | Critical Recall | MAE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['weight']} | {row['f1']} | {row['precision']} | "
            f"{row['recall']} | {row['critical_recall']} | {row['mae']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Study runners
# ---------------------------------------------------------------------------


def run_variance_study(args: argparse.Namespace) -> None:
    """Run the statistical variance study (multi-seed V1 vs SAHL).

    Trains 3 seeds × 2 model variants and prints a summary directing the
    user to aggregate results with ``--mode variance``.

    Args:
        args: Parsed command-line arguments (see :func:`parse_args`).
    """
    seeds = [42, 123, 2026]
    configs: List[ExperimentConfig] = []

    # V1: MSE baseline
    for seed in seeds:
        configs.append(
            ExperimentConfig(
                seed=seed,
                loss_type="mse",
                loss_weight_multiplier=1.0,
                precision_floor=args.precision_floor,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                csv_path=args.csv_path,
                data_root=args.data_root,
                image_size=args.image_size,
                device=args.device,
            )
        )

    # SAHL at the configured weight
    for seed in seeds:
        configs.append(
            ExperimentConfig(
                seed=seed,
                loss_type="weighted_l1",
                loss_weight_multiplier=args.v3_weight,
                precision_floor=args.precision_floor,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                csv_path=args.csv_path,
                data_root=args.data_root,
                image_size=args.image_size,
                device=args.device,
            )
        )

    for config in configs:
        run_single_experiment(config)

    print("\n" + "=" * 70)
    print("  Variance study complete!")
    print("  Next: aggregate with --mode aggregate_variance")
    print("=" * 70 + "\n")


def run_ablation_study(args: argparse.Namespace) -> None:
    """Run the SAHL weight ablation (single seed, 4 weights × 3 seeds).

    Trains 4 weights × 3 seeds = 12 experiments, then aggregates results
    into a Markdown summary table and computes paired significance tests
    comparing 1.0× vs 1.5×.

    Args:
        args: Parsed command-line arguments (see :func:`parse_args`).
    """
    print("Running true multi-seed SAHL weight ablation...")
    for weight in WEIGHTS:
        for seed in SEEDS:
            print(f"  Training weight={weight:.1f} seed={seed}")
            run_single_experiment(
                ExperimentConfig(
                    seed=seed,
                    loss_type="weighted_l1",
                    loss_weight_multiplier=weight,
                    loss_weight_threshold=0.66,
                    critical_recall_threshold=CRITICAL_PREDICTION_THRESHOLD,
                    critical_target_threshold=CRITICAL_TARGET_THRESHOLD,
                    precision_floor=args.precision_floor,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    csv_path=args.csv_path,
                    data_root=args.data_root,
                    image_size=args.image_size,
                    device=args.device,
                )
            )

    summary_rows, raw_metrics = summarize_reports(WEIGHTS, SEEDS)

    method_critical, p_critical = paired_p_value(
        raw_metrics[1.0]["critical_recall"],
        raw_metrics[1.5]["critical_recall"],
    )
    method_mae, p_mae = paired_p_value(
        raw_metrics[1.0]["mae"],
        raw_metrics[1.5]["mae"],
    )

    print("\nPaired significance tests: 1.0x vs 1.5x")
    print(
        f"  Critical Recall: {method_critical} p-value = {p_critical:.6f} "
        f"({'significant' if p_critical < 0.05 else 'not significant'})"
    )
    print(
        f"  MAE: {method_mae} p-value = {p_mae:.6f} "
        f"({'significant' if p_mae < 0.05 else 'not significant'})"
    )

    output_path = Path(args.output_md)
    write_markdown_summary(output_path, summary_rows)
    print(f"\nMarkdown summary written to: {output_path}")
    print(output_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the multi-seed ablation study.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the true multi-seed SAHL weight ablation study or "
            "variance comparison."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ablation",
        choices=["ablation", "variance"],
        help=(
            "Study mode: 'ablation' (4 weights × 3 seeds) or "
            "'variance' (MSE vs SAHL across 3 seeds)."
        ),
    )
    parser.add_argument(
        "--csv_path", type=str, default="labels.csv",
        help="Path to the labels CSV.",
    )
    parser.add_argument(
        "--data_root", type=str, default=".",
        help="Data root directory.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Input image size.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training.",
    )
    parser.add_argument(
        "--precision_floor",
        type=float,
        default=0.0,
        help=(
            "Checkpoint precision floor (default 0.0 so all seeds "
            "complete during ablation)."
        ),
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="multi_seed_ablation_summary.md",
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--v3_weight",
        type=float,
        default=2.5,
        help="SAHL weight multiplier for the variance-study SAHL variant.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: dispatch to the selected study mode."""
    args = parse_args()

    try:
        if args.mode == "variance":
            run_variance_study(args)
        elif args.mode == "ablation":
            run_ablation_study(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
