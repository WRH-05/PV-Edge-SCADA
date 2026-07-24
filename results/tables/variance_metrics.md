# Variance Study Results

Multi-seed study across 3 seeds (42, 123, 2026).
Metrics shown as Mean ± Standard Deviation.

## F1-Score @ threshold 0.65

| Model | F1-Score |
|-------|----------|
| V1 (MSE) | $0.756 \pm 0.013$ |
| V3.1 (SAHL 1.5x) | $0.757 \pm 0.032$ |

## Precision @ threshold 0.65

| Model | Precision |
|-------|-----------|
| V1 (MSE) | $0.856 \pm 0.021$ |
| V3.1 (SAHL 1.5x) | $0.828 \pm 0.035$ |

## Recall @ threshold 0.65

| Model | Recall |
|-------|--------|
| V1 (MSE) | $0.677 \pm 0.032$ |
| V3.1 (SAHL 1.5x) | $0.699 \pm 0.053$ |

## Mean Absolute Error (MAE)

| Model | MAE |
|-------|-----|
| V1 (MSE) | $0.1959 \pm 0.0037$ |
| V3.1 (SAHL 1.5x) | $0.1700 \pm 0.0047$ |

## Critical Recall @ (pred threshold 0.6, target threshold 0.8)

| Model | Critical Recall |
|-------|-----------------|
| V1 (MSE) | $0.784 \pm 0.048$ |
| V3.1 (SAHL 1.5x) | $0.806 \pm 0.046$ |
