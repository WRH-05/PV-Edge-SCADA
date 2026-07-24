#!/usr/bin/env python3
"""Hardware stress-test for ONNX inference on edge devices.

Processes a directory of EL images through a sustained continuous-
inference loop (default 1,588 inferences) and reports:

* **Primary metric:** ONNX-only latency (Mean +/- StdDev), excluding
  disk I/O and preprocessing overhead.
* **Secondary metric:** End-to-end latency (preprocessing + ONNX).
* Peak RSS memory usage measured via ``psutil``.

Designed for the Raspberry Pi 4 but runs on any platform with
``onnxruntime`` and a CPU execution provider.

Usage::

    cd src/inference
    python benchmark_edge.py
    python benchmark_edge.py --loops 2 --onnx_model ../../model/best_sahl_1.5x_final.onnx
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnxruntime as ort

# Allow running from project root: ``python src/inference/benchmark_edge.py``
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from inference_mqtt import preprocess_el_image  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover - optional Pi dependency
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensible defaults (relative to ``src/inference/``)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH: Path = Path("../../model/best_sahl_1.5x_final.onnx")
_DEFAULT_CAPTURES_DIR: Path = Path("../../captures")
_DEFAULT_CANDIDATE_EXTENSIONS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".webp",
)
_DEFAULT_EXPECTED_IMAGE_COUNT: int = 397
_DEFAULT_LOOPS: int = 4          # 4 x 397 = 1,588 inferences
_DEFAULT_WARMUP_RUNS: int = 10
_DEFAULT_IMAGE_SIZE: int = 224
_DEFAULT_CRITICAL_THRESHOLD: float = 0.65


# ===================================================================
# Helpers
# ===================================================================


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_mb(value: Optional[float]) -> str:
    """Format a memory value in MiB, or ``"n/a"`` if unavailable."""
    return "n/a" if value is None else f"{value:.2f}"


def collect_process_rss_mb() -> Optional[float]:
    """Return the current process RSS in MiB, or ``None`` if psutil is absent."""
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def discover_images(captures_dir: Path) -> List[Path]:
    """Return a sorted list of image paths under *captures_dir*."""
    candidates: List[Path] = []
    for ext in _DEFAULT_CANDIDATE_EXTENSIONS:
        candidates.extend(captures_dir.glob(f"*{ext}"))
    return sorted(p for p in candidates if p.is_file())


def percentile(values: List[float], fraction: float) -> float:
    """Compute the *fraction*-th percentile of a numeric list.

    Uses linear interpolation between the two nearest ranks.

    Args:
        values: List of numeric samples.
        fraction: Percentile in [0, 1] (0.95 = P95).

    Returns:
        The interpolated percentile value.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute percentile of an empty series")
    if len(ordered) == 1:
        return ordered[0]
    if fraction <= 0:
        return ordered[0]
    if fraction >= 1:
        return ordered[-1]

    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_csv_rows(
    csv_path: Path,
    fieldnames: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    """Write a list of dicts to a CSV file, creating parent directories."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===================================================================
# Latency summary builder
# ===================================================================


def build_latency_summary(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute aggregate statistics over a list of latency samples.

    Args:
        latencies_ms: Per-inference latency measurements in milliseconds.

    Returns:
        Dictionary with keys ``mean_ms``, ``stddev_ms``, ``median_ms``,
        ``p95_ms``, ``min_ms``, ``max_ms``, ``throughput_fps``.
    """
    if not latencies_ms:
        raise RuntimeError("No latency samples were recorded")

    mean_ms = statistics.fmean(latencies_ms)
    stddev_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    median_ms = statistics.median(latencies_ms)
    p95_ms = percentile(latencies_ms, 0.95)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    total_s = sum(latencies_ms) / 1000.0
    fps = len(latencies_ms) / total_s if total_s > 0 else 0.0

    return {
        "mean_ms": mean_ms,
        "stddev_ms": stddev_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "throughput_fps": fps,
    }


# ===================================================================
# Markdown report
# ===================================================================


def format_markdown_summary(summary: Dict[str, Any]) -> str:
    """Format the benchmark summary as a GitHub-flavoured Markdown table.

    Args:
        summary: Nested dict with ``onnx_primary``, ``end_to_end_secondary``,
            and memory fields.

    Returns:
        Markdown string suitable for writing to a ``.md`` file.
    """
    onnx = summary["onnx_primary"]
    e2e = summary["end_to_end_secondary"]

    lines = [
        "# Edge Sustained Benchmark Summary",
        "",
        "## Run configuration",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Model | {summary['model_path']} |",
        f"| Captures directory | {summary['captures_dir']} |",
        f"| Image count | {summary['image_count']} |",
        f"| Loops | {summary['loops']} |",
        f"| Warmup runs | {summary['warmup_runs']} |",
        f"| Total measured inferences | {summary['measured_inferences']} |",
        "",
        "## Primary latency metric (ONNX-only, excl. disk I/O)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean +/- StdDev (ms) | {onnx['mean_ms']:.2f} +/- {onnx['stddev_ms']:.2f} |",
        f"| Median (ms) | {onnx['median_ms']:.2f} |",
        f"| P95 (ms) | {onnx['p95_ms']:.2f} |",
        f"| Min (ms) | {onnx['min_ms']:.2f} |",
        f"| Max (ms) | {onnx['max_ms']:.2f} |",
        f"| Throughput (FPS) | {onnx['throughput_fps']:.2f} |",
        "",
        "## Secondary latency metric (End-to-end: preprocess + ONNX)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean +/- StdDev (ms) | {e2e['mean_ms']:.2f} +/- {e2e['stddev_ms']:.2f} |",
        f"| Median (ms) | {e2e['median_ms']:.2f} |",
        f"| P95 (ms) | {e2e['p95_ms']:.2f} |",
        f"| Min (ms) | {e2e['min_ms']:.2f} |",
        f"| Max (ms) | {e2e['max_ms']:.2f} |",
        f"| Throughput (FPS) | {e2e['throughput_fps']:.2f} |",
        "",
        "## Memory stability (RSS)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| RSS start (MB) | {fmt_mb(summary['rss_start_mb'])} |",
        f"| RSS after session load (MB) | {fmt_mb(summary['rss_after_session_mb'])} |",
        f"| RSS peak (MB) | {fmt_mb(summary['rss_peak_mb'])} |",
        f"| RSS end (MB) | {fmt_mb(summary['rss_end_mb'])} |",
        f"| Session load delta (MB) | {fmt_mb(summary['rss_session_delta_mb'])} |",
        f"| Start-to-end delta (MB) | {fmt_mb(summary['rss_start_to_end_delta_mb'])} |",
        "",
    ]
    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    """Parse CLI arguments and execute the sustained edge benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Sustained ONNX edge benchmark -- "
            "measures inference latency (ONNX-only & end-to-end), "
            "throughput, and RSS memory footprint."
        ),
    )
    parser.add_argument(
        "--onnx_model",
        type=Path,
        default=_DEFAULT_MODEL_PATH,
        help="Path to the .onnx model file.",
    )
    parser.add_argument(
        "--captures_dir",
        type=Path,
        default=_DEFAULT_CAPTURES_DIR,
        help="Directory containing input EL images.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=_DEFAULT_IMAGE_SIZE,
        help="Square preprocessing edge length in pixels.",
    )
    parser.add_argument(
        "--critical_threshold",
        type=float,
        default=_DEFAULT_CRITICAL_THRESHOLD,
        help="Score > this value means CRITICAL status.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=_DEFAULT_LOOPS,
        help="Number of full passes over the captures directory.",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=_DEFAULT_WARMUP_RUNS,
        help="Number of warmup inferences before measurement.",
    )
    parser.add_argument(
        "--expected_image_count",
        type=int,
        default=_DEFAULT_EXPECTED_IMAGE_COUNT,
        help="Expected number of images (fails early if mismatch).",
    )
    parser.add_argument(
        "--allow_non_expected_count",
        action="store_true",
        help="Skip the strict dataset-size check.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("benchmark_edge_runs.csv"),
        help="Per-inference CSV output path.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("benchmark_edge_summary.csv"),
        help="Single-row aggregate CSV output path.",
    )
    parser.add_argument(
        "--summary_md",
        type=Path,
        default=Path("benchmark_edge_summary.md"),
        help="Markdown summary output path.",
    )
    args = parser.parse_args()

    loops = max(0, args.loops)
    warmup_runs = max(0, args.warmup_runs)

    # Resolve paths relative to this script's directory
    model_path = (_SCRIPT_DIR / args.onnx_model).resolve()
    captures_dir = (_SCRIPT_DIR / args.captures_dir).resolve()
    output_csv = (_SCRIPT_DIR / args.output_csv).resolve()
    summary_csv = (_SCRIPT_DIR / args.summary_csv).resolve()
    summary_md = (_SCRIPT_DIR / args.summary_md).resolve()

    if not model_path.exists():
        logger.error("ONNX model not found: %s", model_path)
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not captures_dir.exists():
        logger.error("Captures directory not found: %s", captures_dir)
        raise FileNotFoundError(f"Captures directory not found: {captures_dir}")

    image_paths = discover_images(captures_dir)
    if not image_paths:
        logger.error("No supported images found in %s", captures_dir)
        raise FileNotFoundError(f"No supported images found in {captures_dir}")

    if not args.allow_non_expected_count and len(image_paths) != args.expected_image_count:
        raise RuntimeError(
            f"Expected exactly {args.expected_image_count} images in {captures_dir}, "
            f"found {len(image_paths)}.  Use --allow_non_expected_count to bypass."
        )

    total_iterations = len(image_paths) * loops
    logger.info("Benchmark config: %d images x %d loops = %d inferences",
                len(image_paths), loops, total_iterations)
    logger.info("Model: %s", model_path)
    logger.info("Warmup runs: %d", warmup_runs)

    # --- Session initialisation & warmup ---
    providers = ["CPUExecutionProvider"]
    rss_start_mb = collect_process_rss_mb()
    session = ort.InferenceSession(str(model_path), providers=providers)
    rss_after_session_mb = collect_process_rss_mb()
    input_name = session.get_inputs()[0].name

    if image_paths:
        warmup_input = preprocess_el_image(str(image_paths[0]), image_size=args.image_size)
        for _ in range(warmup_runs):
            session.run(None, {input_name: warmup_input})

    logger.info("Warmup complete, starting measured runs...")

    # --- Measured runs ---
    rows: List[Dict[str, Any]] = []
    onnx_latencies_ms: List[float] = []
    e2e_latencies_ms: List[float] = []
    failed_count = 0
    rss_peak_mb = rss_after_session_mb

    for loop_index in range(1, loops + 1):
        for image_index, image_path in enumerate(image_paths, start=1):
            global_index = (loop_index - 1) * len(image_paths) + image_index
            rss_before_run_mb = collect_process_rss_mb()

            score: Optional[float] = None
            status = "ERROR"
            error = ""

            e2e_start = time.perf_counter()
            try:
                # Preprocess once, reuse for both measurements
                e2e_input = preprocess_el_image(
                    str(image_path), image_size=args.image_size
                )
                # Full e2e: preprocess + ONNX
                session.run(None, {input_name: e2e_input})
                e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000.0

                # ONNX-only: exclude disk I/O and preprocessing
                onnx_start = time.perf_counter()
                outputs = session.run(None, {input_name: e2e_input})
                onnx_latency_ms = (time.perf_counter() - onnx_start) * 1000.0

                score = float(outputs[0][0, 0])
                status = "CRITICAL" if score > args.critical_threshold else "OK"
                onnx_latencies_ms.append(onnx_latency_ms)
                e2e_latencies_ms.append(e2e_latency_ms)

            except Exception as exc:
                e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000.0
                onnx_latency_ms = 0.0
                error = str(exc)
                failed_count += 1
                logger.warning("Inference %d failed: %s", global_index, exc)

            rss_after_run_mb = collect_process_rss_mb()
            if rss_after_run_mb is not None:
                rss_peak_mb = (
                    rss_after_run_mb
                    if rss_peak_mb is None
                    else max(rss_peak_mb, rss_after_run_mb)
                )

            rows.append({
                "timestamp_utc": utc_now(),
                "global_index": global_index,
                "loop_index": loop_index,
                "image_index": image_index,
                "image_path": str(image_path),
                "severity_score": None if score is None else round(score, 6),
                "status": status,
                "onnx_latency_ms": round(onnx_latency_ms, 4),
                "end_to_end_latency_ms": round(e2e_latency_ms, 4),
                "rss_before_run_mb": (
                    None if rss_before_run_mb is None
                    else round(rss_before_run_mb, 4)
                ),
                "rss_after_run_mb": (
                    None if rss_after_run_mb is None
                    else round(rss_after_run_mb, 4)
                ),
                "rss_delta_mb": (
                    None
                    if (rss_before_run_mb is None or rss_after_run_mb is None)
                    else round(rss_after_run_mb - rss_before_run_mb, 4)
                ),
                "error": error,
            })

            if global_index % 100 == 0 or global_index == total_iterations:
                logger.info(
                    "Progress: %d/%d (failed: %d)",
                    global_index, total_iterations, failed_count,
                )

    if not onnx_latencies_ms:
        raise RuntimeError("No successful inferences were recorded")

    rss_end_mb = collect_process_rss_mb()

    # --- Compute deltas ---
    rss_session_delta_mb: Optional[float] = None
    if rss_start_mb is not None and rss_after_session_mb is not None:
        rss_session_delta_mb = rss_after_session_mb - rss_start_mb

    rss_start_to_end_delta_mb: Optional[float] = None
    if rss_start_mb is not None and rss_end_mb is not None:
        rss_start_to_end_delta_mb = rss_end_mb - rss_start_mb

    # --- Summary ---
    onnx_summary = build_latency_summary(onnx_latencies_ms)
    e2e_summary = build_latency_summary(e2e_latencies_ms)

    summary: Dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "model_path": str(model_path),
        "captures_dir": str(captures_dir),
        "image_count": len(image_paths),
        "loops": loops,
        "warmup_runs": warmup_runs,
        "measured_inferences": len(onnx_latencies_ms),
        "failed_inferences": failed_count,
        # ONNX primary
        "onnx_mean_latency_ms": onnx_summary["mean_ms"],
        "onnx_stddev_latency_ms": onnx_summary["stddev_ms"],
        "onnx_median_latency_ms": onnx_summary["median_ms"],
        "onnx_p95_latency_ms": onnx_summary["p95_ms"],
        "onnx_min_latency_ms": onnx_summary["min_ms"],
        "onnx_max_latency_ms": onnx_summary["max_ms"],
        "onnx_throughput_fps": onnx_summary["throughput_fps"],
        # E2E secondary
        "e2e_mean_latency_ms": e2e_summary["mean_ms"],
        "e2e_stddev_latency_ms": e2e_summary["stddev_ms"],
        "e2e_median_latency_ms": e2e_summary["median_ms"],
        "e2e_p95_latency_ms": e2e_summary["p95_ms"],
        "e2e_min_latency_ms": e2e_summary["min_ms"],
        "e2e_max_latency_ms": e2e_summary["max_ms"],
        "e2e_throughput_fps": e2e_summary["throughput_fps"],
        # Memory
        "rss_start_mb": rss_start_mb,
        "rss_after_session_mb": rss_after_session_mb,
        "rss_peak_mb": rss_peak_mb,
        "rss_end_mb": rss_end_mb,
        "rss_session_delta_mb": rss_session_delta_mb,
        "rss_start_to_end_delta_mb": rss_start_to_end_delta_mb,
    }

    # --- Write outputs ---
    write_csv_rows(
        output_csv,
        [
            "timestamp_utc", "global_index", "loop_index", "image_index",
            "image_path", "severity_score", "status",
            "onnx_latency_ms", "end_to_end_latency_ms",
            "rss_before_run_mb", "rss_after_run_mb", "rss_delta_mb", "error",
        ],
        rows,
    )
    logger.info("Per-inference CSV written to %s", output_csv)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    logger.info("Summary CSV written to %s", summary_csv)

    md_doc = {
        "model_path": summary["model_path"],
        "captures_dir": summary["captures_dir"],
        "image_count": summary["image_count"],
        "loops": summary["loops"],
        "warmup_runs": summary["warmup_runs"],
        "measured_inferences": summary["measured_inferences"],
        "onnx_primary": onnx_summary,
        "end_to_end_secondary": e2e_summary,
        "rss_start_mb": summary["rss_start_mb"],
        "rss_after_session_mb": summary["rss_after_session_mb"],
        "rss_peak_mb": summary["rss_peak_mb"],
        "rss_end_mb": summary["rss_end_mb"],
        "rss_session_delta_mb": summary["rss_session_delta_mb"],
        "rss_start_to_end_delta_mb": summary["rss_start_to_end_delta_mb"],
    }
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(format_markdown_summary(md_doc), encoding="utf-8")
    logger.info("Markdown summary written to %s", summary_md)

    # Print to console
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Benchmark stopped by user")
        sys.exit(0)
