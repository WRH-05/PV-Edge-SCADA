# src/analysis — Rigour & Evaluation

Statistical analysis and publication-quality visualisation for the EL
defect severity regression pipeline.

## Files

| File | Purpose |
|------|---------|
| `multi_seed_ablation.py` | Run the 3-seed, 4-weight SAHL ablation study and output Markdown summary tables with paired significance tests |
| `generate_plots.py` | Generate Precision-Recall, F1-Threshold, Confusion Matrix, ROC, and Error Residual plots using `matplotlib` and `scikit-learn` |

## Quick Start (3 Steps)

### 1. Run the multi-seed ablation study

```powershell
python src/analysis/multi_seed_ablation.py `
    --mode ablation --csv_path labels.csv --data_root . `
    --epochs 20 --output_md ablation_results.md
```

This trains 4 SAHL weight configurations (1.0×, 1.5×, 2.5×, 5.0×) across
3 random seeds (42, 123, 2026) and writes a Markdown summary table with
Mean ± Std metrics and paired significance tests (1.0× vs 1.5×).

For the variance comparison (MSE baseline vs SAHL across 3 seeds):

```powershell
python src/analysis/multi_seed_ablation.py `
    --mode variance --v3_weight 2.5 --epochs 20
```

### 2. Generate evaluation plots

```powershell
python src/analysis/generate_plots.py `
    --v1_csv testCsv/test_split_report_v1_onnx.csv `
    --v3_csv testCsv/test_split_report_v3_1_goldilocks_onnx.csv `
    --output_dir results/ --formats png pdf
```

Produces five figures and a `metrics_summary.txt` in the specified output
directory:
- `pr_curve_v1_vs_v3_1.{png,pdf}` — Precision-Recall curve
- `roc_curve_v1_vs_v3_1.{png,pdf}` — ROC curve
- `f1_vs_threshold.{png,pdf}` — F1 score vs decision threshold
- `confusion_matrices_at_operating_threshold.{png,pdf}` — side-by-side confusion matrices
- `error_distribution_comparison.{png,pdf}` — residual histograms

### 3. Customise plot appearance

```powershell
python src/analysis/generate_plots.py `
    --v1_csv my_v1.csv --v3_csv my_v3.csv `
    --v1_name "MSE Baseline" --v3_name "SAHL (2.5x)" `
    --target_threshold 0.65 --operating_threshold 0.65 `
    --output_dir results/ --dpi 400
```

## Notes

- All paths are CLI arguments with relative defaults.
- The ablation study uses `sys.executable` (not a hardcoded venv path)
  for cross-platform subprocess calls.
- Plots follow a consistent visual style suitable for direct inclusion in
  IEEE/ACM manuscripts.
