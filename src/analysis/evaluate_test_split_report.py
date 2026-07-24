"""Generate a per-sample test-split evaluation report CSV from an ONNX model.

This script is the third stage of the multi-seed ablation pipeline (train →
export → report).  It loads an exported ``.onnx`` model, runs inference on
the held-out test split produced by :func:`src.training.dataset.create_dataloaders`,
and writes a CSV with columns ``image_path``, ``target``, ``prediction``,
``abs_error``, and ``squared_error``.

Typical usage::

    python evaluate_test_split_report.py \\
        --csv_path labels.csv --data_root . --seed 42 \\
        --onnx_model models/onnx/best_model.onnx \\
        --output_csv testCsv/report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from tqdm import tqdm

from src.training.dataset import create_dataloaders, set_seed

# ---------------------------------------------------------------------------
# ImageNet normalisation (must match training transforms)
# ---------------------------------------------------------------------------
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def run_onnx_inference(
    session: ort.InferenceSession,
    image_batch: torch.Tensor,
) -> np.ndarray:
    """Run a batch of preprocessed images through the ONNX session.

    The input tensor is expected to be in CHW format with ImageNet
    normalisation already applied (i.e. the output of the training
    DataLoader transform pipeline).

    Args:
        session: An initialised ``onnxruntime.InferenceSession``.
        image_batch: Tensor of shape ``(N, 3, H, W)``.

    Returns:
        Predictions as a float64 array of shape ``(N,)``.
    """
    input_name = session.get_inputs()[0].name
    # Convert to numpy in NCHW layout (matches ONNX expected input)
    np_input = image_batch.cpu().numpy().astype(np.float32)
    outputs = session.run(None, {input_name: np_input})
    return outputs[0].flatten().astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_report(
    csv_path: str,
    data_root: str,
    seed: int,
    image_size: int,
    onnx_model: str,
    output_csv: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> None:
    """Generate a per-sample test-set evaluation report.

    Creates a 70/15/15 stratified split with the given seed, runs ONNX
    inference on the test portion, and writes the results to ``output_csv``.

    Args:
        csv_path: Path to the labels CSV.
        data_root: Root directory for resolving image paths.
        seed: Random seed (must match the training seed).
        image_size: Input image dimension in pixels.
        onnx_model: Path to the ``.onnx`` file.
        output_csv: Destination path for the report CSV.
        batch_size: Mini-batch size for inference.
        num_workers: DataLoader worker processes.
    """
    set_seed(seed)

    onnx_path = Path(onnx_model)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # Re-create the same train/val/test split used during training
    # We only use the test loader here.
    _train_loader, _val_loader, test_loader, split_counts = create_dataloaders(
        csv_path=csv_path,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        seed=seed,
    )

    print(f"Split counts: {split_counts}")
    print(f"Test samples: {split_counts['test']}")

    # Load ONNX session
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    # Run inference over the test split
    image_paths: list[str] = []
    all_targets: list[float] = []
    all_predictions: list[float] = []

    # The DataLoader uses a sampler on the training set; test_loader is
    # sequential (shuffle=False), so we can walk the underlying dataset.
    test_dataset = test_loader.dataset
    for idx in tqdm(range(len(test_dataset)), desc="Inference"):
        image_tensor, target = test_dataset[idx]
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0)
        pred = run_onnx_inference(session, image_batch)

        item_path, _ = test_dataset.items[idx]
        image_paths.append(str(item_path))
        all_targets.append(float(target.item()))
        all_predictions.append(float(pred[0]))

    # Build DataFrame
    targets_arr = np.array(all_targets, dtype=np.float64)
    preds_arr = np.array(all_predictions, dtype=np.float64)
    abs_error = np.abs(preds_arr - targets_arr)
    squared_error = (preds_arr - targets_arr) ** 2

    df = pd.DataFrame({
        "image_path": image_paths,
        "target": targets_arr,
        "prediction": preds_arr,
        "abs_error": abs_error,
        "squared_error": squared_error,
    })

    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print summary
    mae = float(np.mean(abs_error))
    mse = float(np.mean(squared_error))
    print(f"Report written to: {output_path}")
    print(f"  Samples: {len(df)}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")


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
            "Generate a per-sample test-split evaluation report by running "
            "ONNX inference on the held-out test set."
        )
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
        "--seed", type=int, default=42,
        help="Random seed (must match training seed).",
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Input image dimension.",
    )
    parser.add_argument(
        "--onnx_model", type=str, required=True,
        help="Path to the .onnx model file.",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Destination path for the report CSV.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Mini-batch size for inference.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader worker processes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_report(
        csv_path=args.csv_path,
        data_root=args.data_root,
        seed=args.seed,
        image_size=args.image_size,
        onnx_model=args.onnx_model,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
