# src/training — Core ML Pipeline

Transfer-learning training pipeline for EL solar cell defect severity
regression using a ResNet18 backbone with a warmup-to-unfreeze schedule.

## Files

| File | Purpose |
|------|---------|
| `dataset.py` | CSV loading, 5×5 median filtering, pseudo-RGB conversion, ImageNet normalisation, bucketed stratified 70/15/15 splitting |
| `losses.py` | `SafetyAwareAsymmetricHuberLoss` (SAHL) with two-condition gate (y ≥ τ and ŷ < τ), `WeightedMSELoss`, and `build_loss` factory |
| `train.py` | Main training orchestration with warmup-to-unfreeze scheduling and `.pth` checkpoint save |
| `export_onnx.py` | Convert trained `.pth` models to `.onnx` (Opset 18) |

## Quick Start (3 Steps)

### 1. Train a model

```powershell
python -m src.training.train `
    --csv_path labels.csv --data_root . `
    --epochs 20 --batch_size 32 --loss_type sahl `
    --checkpoint_path best_model.pth
```

Key options:
- `--loss_type`: `smoothl1` (symmetric Huber), `mse`, `weighted_l1`, `weighted_mse`, or `sahl` (two-condition SAHL with optional Huber beta)
- `--loss_weight_multiplier`: critical-sample penalty (default 2.5)
- `--loss_weight_threshold`: decision boundary τ for the two-condition gate (default 0.70, matches manuscript)
- `--warmup_epochs`: epochs before backbone unfreezing (default 2)

### 2. Export to ONNX

```powershell
python -m src.training.export_onnx `
    --checkpoint best_model.pth --onnx_output best_model.onnx
```

The exported model targets **Opset 18** (compatible with ONNX Runtime ≥ 1.14).

### 3. Inspect the dataset pipeline

```powershell
python -m src.training.dataset `
    --csv_path labels.csv --data_root .
```

Prints split counts and a sample batch shape to verify the pipeline is
configured correctly before training.

## Notes

- The median filter kernel is **5×5** (updated from 3×3 for improved
  denoising per the published methodology).
- All paths are CLI arguments with relative defaults — no hardcoded
  absolute paths.
- The `SAHL` loss type with `--huber_beta 0.0` (default) behaves
  identically to the original `weighted_l1` loss for backward
  compatibility with existing ablation results.
