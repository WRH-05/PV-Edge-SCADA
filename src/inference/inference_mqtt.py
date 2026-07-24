#!/usr/bin/env python3
"""Edge inference worker for EL defect severity regression via ONNX Runtime.

Loads an ONNX model exported from a PyTorch ResNet-based regressor,
preprocesses electroluminescence images, runs inference, evaluates a
critical safety threshold (>= 0.65), and publishes structured JSON
payloads over MQTT.

Usage:
    python inference_mqtt.py --image_path /path/to/el_image.png
    python inference_mqtt.py --image_path ... --mqtt_enable
    MQTT_BROKER=10.0.0.5 python inference_mqtt.py --image_path ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalisation constants used during PyTorch training
# ---------------------------------------------------------------------------
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Sensible defaults (relative to the ``src/inference/`` package directory)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH: str = "../../model/best_sahl_1.5x_final.onnx"
_DEFAULT_IMAGE_SIZE: int = 224
_DEFAULT_CRITICAL_THRESHOLD: float = 0.65
_DEFAULT_MQTT_BROKER: str = "localhost"
_DEFAULT_MQTT_PORT: int = 1883
_DEFAULT_MQTT_TOPIC: str = "pv/inspection/severity"


# ===================================================================
# Image preprocessing
# ===================================================================


def preprocess_el_image(image_path: str, image_size: int = 224) -> np.ndarray:
    """Preprocess an EL image for ONNX inference.

    Pipeline:
        1. Read as grayscale.
        2. Denoise with a 5×5 median blur.
        3. Stack to 3-channel RGB (repeated grayscale).
        4. Resize to ``image_size x image_size`` (bilinear interpolation).
        5. Normalise to [0, 1], then apply ImageNet mean/std.
        6. Transpose HWC -> CHW and add batch dimension.

    Args:
        image_path: Filesystem path to the input image.
        image_size: Target square edge length in pixels.

    Returns:
        A float32 array of shape ``(1, 3, image_size, image_size)``
        ready for ONNX inference.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    denoised = cv2.medianBlur(gray, 5)

    # Convert single-channel gray to 3-channel pseudo-RGB
    rgb = np.stack([denoised, denoised, denoised], axis=-1)

    # Match torchvision Resize default behaviour (bilinear interpolation)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # Normalise using the same transform applied during training
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # Convert HWC -> CHW, then add batch dimension
    chw = np.transpose(normalized, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0).astype(np.float32)
    return batched


# ===================================================================
# ONNX inference
# ===================================================================


def infer_severity_score(
    onnx_model_path: str,
    image_path: str,
    image_size: int = 224,
    session: Optional[ort.InferenceSession] = None,
) -> float:
    """Run ONNX inference and return a scalar defect severity score.

    If no session is supplied a new ``InferenceSession`` is created
    on-the-fly using the CPU execution provider.  Reusing a pre-warmed
    session across multiple calls is strongly recommended for low-latency
    or batch workloads.

    Args:
        onnx_model_path: Path to the ``.onnx`` model file.
        image_path: Path to the input image.
        image_size: Square edge length for preprocessing.
        session: An existing ``ort.InferenceSession``, or ``None``.

    Returns:
        A float in [0, 1] representing the defect severity level
        (higher = more critical).
    """
    if session is None:
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    model_input = preprocess_el_image(image_path=image_path, image_size=image_size)

    outputs = session.run(None, {input_name: model_input})
    severity_score = float(outputs[0][0, 0])
    return severity_score


# ===================================================================
# Payload construction & threshold logic
# ===================================================================


def build_payload(
    panel_id: str,
    pad_id: str,
    robot_id: str,
    model_version: str,
    severity_score: float,
    critical_threshold: float = 0.65,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a JSON-serialisable inspection payload.

    The safety threshold is evaluated as **> threshold** (i.e. strictly
    greater than).  A score of exactly 0.65 is **not** considered
    critical by default.

    Args:
        panel_id: Identifier of the solar panel.
        pad_id: Identifier of the individual pad/cell.
        robot_id: Identifier of the inspection robot.
        model_version: Version string of the deployed ONNX model.
        severity_score: Raw scalar output from the regression model.
        critical_threshold: Decision boundary for CRITICAL status.
        image_path: Optional filesystem path recorded for traceability.

    Returns:
        Dictionary ready for JSON serialisation and MQTT publishing.
    """
    status: str = "CRITICAL" if severity_score > critical_threshold else "OK"
    return {
        "panel_id": panel_id,
        "pad_id": pad_id,
        "robot_id": robot_id,
        "model_version": model_version,
        "severity_score": round(severity_score, 4),
        "status": status,
        "image_path": image_path,
    }


# ===================================================================
# MQTT publishing
# ===================================================================


def publish_mqtt(
    payload: Dict[str, Any],
    broker_host: str,
    broker_port: int,
    topic: str,
) -> None:
    """Publish a JSON payload to an MQTT broker.

    The connection is opened, the message is published with QoS 0
    (fire-and-forget), and the client disconnects immediately.

    Args:
        payload: JSON-serialisable dictionary.
        broker_host: MQTT broker hostname or IP address.
        broker_port: MQTT broker TCP port.
        topic: MQTT topic string.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker_host, broker_port, keepalive=30)
    client.publish(topic, json.dumps(payload), qos=0, retain=False)
    client.disconnect()


# ===================================================================
# CLI entry point
# ===================================================================


def _resolve_path(raw: str) -> Path:
    """Resolve a possibly-relative path against the script's directory."""
    script_dir = Path(__file__).resolve().parent
    return (script_dir / raw).resolve()


def main() -> None:
    """Parse CLI arguments, run inference, and optionally publish via MQTT."""
    parser = argparse.ArgumentParser(
        description="ONNX Runtime EL inference with optional MQTT publishing."
    )

    # ---- Model & image ----
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("ONNX_MODEL_PATH", _DEFAULT_MODEL_PATH),
        help="Path to the .onnx model file (env: ONNX_MODEL_PATH).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input EL image.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=_DEFAULT_IMAGE_SIZE,
        help="Square preprocessing edge length in pixels.",
    )

    # ---- Inspection metadata ----
    parser.add_argument("--panel_id", type=str, default="panel_A")
    parser.add_argument("--pad_id", type=str, default="simulated_pad_01")
    parser.add_argument("--robot_id", type=str, default="robot_01")
    parser.add_argument("--model_version", type=str, default="onnx_v1")

    # ---- Safety threshold ----
    parser.add_argument(
        "--critical_threshold",
        type=float,
        default=_DEFAULT_CRITICAL_THRESHOLD,
        help="Score > this value means CRITICAL status.",
    )

    # ---- MQTT (env-var aware) ----
    parser.add_argument(
        "--mqtt_enable",
        action="store_true",
        default=os.getenv("MQTT_ENABLE", "").lower() in ("1", "true", "yes"),
        help="Publish the result to MQTT.  Also set MQTT_ENABLE=1.",
    )
    parser.add_argument(
        "--mqtt_broker",
        type=str,
        default=os.getenv("MQTT_BROKER", _DEFAULT_MQTT_BROKER),
        help="MQTT broker hostname or IP (env: MQTT_BROKER).",
    )
    parser.add_argument(
        "--mqtt_port",
        type=int,
        default=int(os.getenv("MQTT_PORT", str(_DEFAULT_MQTT_PORT))),
        help="MQTT broker TCP port (env: MQTT_PORT).",
    )
    parser.add_argument(
        "--mqtt_topic",
        type=str,
        default=os.getenv("MQTT_TOPIC", _DEFAULT_MQTT_TOPIC),
        help="MQTT topic to publish to (env: MQTT_TOPIC).",
    )

    args = parser.parse_args()

    # Resolve relative paths against the script directory
    model_path = _resolve_path(args.model_path)
    image_path = _resolve_path(args.image_path)

    if not model_path.exists():
        logger.error("ONNX model not found: %s", model_path)
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not image_path.exists():
        logger.error("Input image not found: %s", image_path)
        raise FileNotFoundError(f"Input image not found: {image_path}")

    logger.info("Loading ONNX session from %s", model_path)
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)

    logger.info("Running inference on %s", image_path)
    score = infer_severity_score(
        onnx_model_path=str(model_path),
        image_path=str(image_path),
        image_size=args.image_size,
        session=session,
    )

    payload = build_payload(
        panel_id=args.panel_id,
        pad_id=args.pad_id,
        robot_id=args.robot_id,
        model_version=args.model_version,
        severity_score=score,
        critical_threshold=args.critical_threshold,
        image_path=str(image_path),
    )

    logger.info(
        "Inference complete -- score=%.4f, status=%s",
        payload["severity_score"],
        payload["status"],
    )
    print(json.dumps(payload, indent=2))

    if args.mqtt_enable:
        logger.info(
            "Publishing to MQTT broker %s:%d on topic '%s'",
            args.mqtt_broker,
            args.mqtt_port,
            args.mqtt_topic,
        )
        try:
            publish_mqtt(
                payload=payload,
                broker_host=args.mqtt_broker,
                broker_port=args.mqtt_port,
                topic=args.mqtt_topic,
            )
            logger.info("MQTT publish succeeded")
        except Exception:
            logger.exception("MQTT publish failed (non-blocking)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
