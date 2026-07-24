# Edge-AI Asymmetric Loss Optimization for PV Inspection

This repository contains the data, models, and systems architecture for the paper: **"Edge-AI Severity Regression and Asymmetric Loss for Controlled PV Electroluminescence Benchmarking and SCADA Alerting."**

## Project Overview
Current drone-based photovoltaic (PV) inspections suffer from wind-induced motion blur during 15-second nighttime Electroluminescence (EL) exposures. This project proposes a cyber-physical architecture utilizing a theoretical frame-crawler to define hardware constraints, paired with a fully validated **Edge-AI and IoT SCADA pipeline**.

### Core Innovation: Safety-Aware Asymmetric Huber Loss (SAHL)
Standard Mean Squared Error (MSE) models optimize for average error, leading to the dangerous under-prediction of severe, safety-critical micro-cracks. We introduce **SAHL**, a cost-sensitive regression objective. By applying a 1.5x/2.5x penalty to critical underestimation events, the model mathematically prioritizes high-consequence failure capture.

### Key Performance Metrics (Multi-Seed Validation)
* **Critical Defect Recall:** Improved to 80.6% (vs. 78.4% baseline) using a 1.5x asymmetry weight (84.6% with 2.5x weight).
* **Edge Latency (Raspberry Pi 4):** 190.32 ± 27.93 ms.
* **Memory Footprint:** 128.98 MB Peak RSS (Stable over 1,588 continuous inferences).

## Repository Structure
- `manuscript/`: Contains the final compiled PDF and LaTeX source code.
- `models/`: Contains the ONNX-quantized and PyTorch models for SAHL weights (1.0x, 1.5x, 2.5x).
- `data/`: Contains benchmark summaries, multi-seed ablation metrics, and edge hardware stress-test logs.
- `src/`: Contains the Node-RED flow JSON and PostgreSQL/Supabase alert-state trigger logic.

## Data Availability
This study utilizes the public ZAE Bayern EL dataset for all model training and validation. The raw images are not hosted in this repository; only the derived metrics, architectural codebase, and dashboard artifacts are provided.

## Disclaimer
This work was independently developed by the author using personal hardware and public datasets. The author acknowledges the physical space provided by the Centre de Développement des Énergies Renouvelables (CDER) during the preliminary stages of this study.