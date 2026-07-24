"""Edge ONNX inference package for EL defect severity regression.

This package provides:
- ``inference_mqtt``: Core inference worker with ONNX model loading,
  image preprocessing, severity scoring, and MQTT publishing.
- ``benchmark_edge``: Sustained hardware stress-test measuring
  inference latency, throughput, and RSS memory footprint.
"""
