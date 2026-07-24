# Multi-Seed SAHL Weight Ablation Summary

Metrics are reported as Mean ± Std across seeds [42, 123, 2026].
General metrics use threshold 0.65; critical recall uses pred >= 0.6 and target >= 0.8.

| Weight | F1-Score | Precision | Recall | Critical Recall | MAE |
| --- | --- | --- | --- | --- | --- |
| 1.0x | 0.774 ± 0.008 | 0.854 ± 0.065 | 0.712 ± 0.042 | 0.815 ± 0.040 | 0.1734 ± 0.0125 |
| 1.5x | 0.757 ± 0.032 | 0.828 ± 0.035 | 0.699 ± 0.053 | 0.806 ± 0.046 | 0.1700 ± 0.0047 |
| 2.5x | 0.758 ± 0.020 | 0.757 ± 0.018 | 0.760 ± 0.049 | 0.846 ± 0.028 | 0.1898 ± 0.0122 |
| 5.0x | 0.551 ± 0.129 | 0.423 ± 0.181 | 0.896 ± 0.098 | 0.944 ± 0.073 | 0.4524 ± 0.1821 |