# Edge Sustained Benchmark Summary

## Run configuration

| Metric | Value |
| --- | ---: |
| Model | /home/chrome/CderPiCamService/model/best_sahl_1.5x_final.onnx |
| Captures directory | /home/chrome/CderPiCamService/captures |
| Image count | 397 |
| Loops | 4 |
| Warmup runs | 10 |
| Total measured inferences | 1588 |

## Primary latency metric (ONNX-only)

| Metric | Value |
| --- | ---: |
| Mean ± StdDev (ms) | 190.32 +/- 27.93 |
| Median (ms) | 177.63 |
| P95 (ms) | 238.77 |
| Min (ms) | 157.47 |
| Max (ms) | 296.75 |
| Throughput (FPS) | 5.25 |

## Secondary latency metric (End-to-end)

| Metric | Value |
| --- | ---: |
| Mean ± StdDev (ms) | 205.98 +/- 29.50 |
| Median (ms) | 192.37 |
| P95 (ms) | 254.03 |
| Min (ms) | 167.61 |
| Max (ms) | 323.44 |
| Throughput (FPS) | 4.85 |

## Memory stability (RSS)

| Metric | Value |
| --- | ---: |
| RSS start (MB) | 61.59 |
| RSS after session load (MB) | 69.40 |
| RSS peak (MB) | 128.98 |
| RSS end (MB) | 127.80 |
| Session load delta (MB) | 7.82 |
| Start to end delta (MB) | 66.22 |
