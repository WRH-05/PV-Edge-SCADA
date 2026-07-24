# Edge ONNX Inference & Benchmarking

Refactored inference worker and sustained stress-test scripts for Raspberry Pi 4
deployment of an SAHL (Safety-Aware Asymmetric Huber Loss) severity regression
model.  Part of the **CDER Pi Camera Service** project.

## Files

| File | Purpose |
|------|---------|
| `inference_mqtt.py` | Single-image inference worker with optional MQTT publishing |
| `benchmark_edge.py` | Sustained hardware stress-test (1,588 inferences) |
| `requirements_edge.txt` | Minimal Pi runtime dependencies |

## Prerequisites

- Python 3.9 or later
- Virtual environment (recommended)
- The ONNX model (e.g., `sahl_1.5x.onnx`) and its external-data companion
  (`.onnx.data`) placed in `models/policy_1.5x_low_error/` (or equivalent) at the project root.

## Quick Start

```bash
# 1. From the project root, create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# 2. Install minimal dependencies
pip install -r src/inference/requirements_edge.txt

# 3. Verify the installation
python -m py_compile src/inference/inference_mqtt.py
python -m py_compile src/inference/benchmark_edge.py
```

---

## `inference_mqtt.py` — Edge Inference Worker

Runs a single EL image through the ONNX model and prints a JSON payload.
Optionally publishes the result to an MQTT broker.

### Basic usage

```bash
cd src/inference
python inference_mqtt.py --image_path ../../captures/cell0041.png
```

### With MQTT publishing enabled

**Via environment variables (recommended):**

```bash
export MQTT_BROKER=10.0.0.5
export MQTT_PORT=1883
export MQTT_TOPIC=pv/inspection/severity
export MQTT_ENABLE=1

python inference_mqtt.py --image_path ../../captures/cell0041.png
```

**Via CLI flags (override env vars):**

```bash
python inference_mqtt.py \
  --image_path ../../captures/cell0041.png \
  --mqtt_enable \
  --mqtt_broker 10.0.0.5 \
  --mqtt_port 1883 \
  --mqtt_topic pv/inspection/severity
```

### All CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | `../../models/policy_1.5x_low_error/sahl_1.5x.onnx` | Path to `.onnx` file (env: `ONNX_MODEL_PATH`) |
| `--image_path` | *required* | Input EL image |
| `--image_size` | `224` | Preprocessing edge length |
| `--panel_id` | `panel_A` | Panel identifier |
| `--pad_id` | `simulated_pad_01` | Pad / cell identifier |
| `--robot_id` | `robot_01` | Inspection robot identifier |
| `--model_version` | `onnx_v1` | Model version string |
| `--critical_threshold` | `0.65` | Score > this → CRITICAL |
| `--mqtt_enable` | `False` | Publish to MQTT (env: `MQTT_ENABLE=1`) |
| `--mqtt_broker` | `localhost` | Broker hostname (env: `MQTT_BROKER`) |
| `--mqtt_port` | `1883` | Broker TCP port (env: `MQTT_PORT`) |
| `--mqtt_topic` | `pv/inspection/severity` | MQTT topic (env: `MQTT_TOPIC`) |

### Expected output

```json
{
  "panel_id": "panel_A",
  "pad_id": "simulated_pad_01",
  "robot_id": "robot_01",
  "model_version": "onnx_v1",
  "severity_score": 0.8321,
  "status": "CRITICAL",
  "image_path": "/absolute/path/to/captures/cell0041.png"
}
```

---

## `benchmark_edge.py` — Sustained Edge Benchmark

Processes a directory of EL images through a sustained continuous-inference loop
(default: **1,588 inferences** = 397 images × 4 loops) and reports:

- **Primary:** ONNX-only latency (Mean ± StdDev), excluding disk I/O.
- **Secondary:** End-to-end latency (preprocessing + ONNX).
- **Peak RSS** memory usage throughout the run.

### Basic usage (with defaults)

```bash
cd src/inference
python benchmark_edge.py
```

### Custom directories / loop count

```bash
python benchmark_edge.py \
  --onnx_model ../../model/best_sahl_1.5x_final.onnx \
  --captures_dir ../../captures \
  --loops 4 \
  --warmup_runs 10
```

### All CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--onnx_model` | `../../models/policy_1.5x_low_error/sahl_1.5x.onnx` | Path to `.onnx` file |
| `--captures_dir` | `../../captures` | Directory of input images |
| `--image_size` | `224` | Preprocessing edge length |
| `--critical_threshold` | `0.65` | Score > this → CRITICAL |
| `--loops` | `4` | Full passes over captures dir |
| `--warmup_runs` | `10` | Warmup inferences before timing |
| `--expected_image_count` | `397` | Fail if count differs (use `--allow_non_expected_count` to bypass) |
| `--allow_non_expected_count` | `False` | Skip strict count check |
| `--output_csv` | `benchmark_edge_runs.csv` | Per-inference CSV |
| `--summary_csv` | `benchmark_edge_summary.csv` | Aggregate summary CSV |
| `--summary_md` | `benchmark_edge_summary.md` | Markdown report |

### Output

Three files are produced:
1. **Per-inference CSV** — one row per inference with timestamps, latencies, RSS, scores.
2. **Summary CSV** — single-row aggregate statistics.
3. **Markdown summary** — formatted table printed to stdout and saved to disk.

---

## Environment Variables

Both scripts honour the following environment variables.
CLI arguments always take precedence over environment variables.

| Variable | Used by | Default |
|----------|---------|---------|
| `ONNX_MODEL_PATH` | `inference_mqtt.py` | `../../models/policy_1.5x_low_error/sahl_1.5x.onnx` |
| `MQTT_ENABLE` | `inference_mqtt.py` | (disabled) |
| `MQTT_BROKER` | `inference_mqtt.py` | `localhost` |
| `MQTT_PORT` | `inference_mqtt.py` | `1883` |
| `MQTT_TOPIC` | `inference_mqtt.py` | `pv/inspection/severity` |

---

## Dependencies

Installed via `requirements_edge.txt`:

```
numpy>=1.21
opencv-python-headless>=4.5
onnxruntime>=1.14
paho-mqtt>=2.0
psutil>=5.9
```

**Note:** This list intentionally excludes heavy packages such as PyTorch,
TorchVision, Matplotlib, and SciPy — they are not required for inference and
would bloat the Raspberry Pi deployment.
