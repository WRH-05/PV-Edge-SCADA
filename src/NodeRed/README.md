# Node-RED Supabase Ingest Flow

This folder contains an importable Node-RED flow that:

1. Subscribes to MQTT topic pv/inspection/severity
2. Validates and maps payload to Supabase inspections table shape
3. Inserts row into inspections
4. Calls Supabase RPC evaluate_pad_alert
5. Routes OPEN vs non-OPEN alert states for dashboard and notifications

## Files

- supabase_ingest_flow.json: Minimal ingest-only flow (MQTT -> Supabase -> alert routing)
- full_cder_supabase_dashboard_flow.json: Full dashboard + Supabase ingest + DB alert count polling

## Required Node-RED Environment Variables

Set these in your Node-RED runtime before deploying the flow:

- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY

## Import Steps

1. Open Node-RED editor
2. Menu -> Import
3. Paste contents of supabase_ingest_flow.json or select file
4. Update MQTT broker node host and auth
5. Deploy

## Test

Publish one test message to MQTT topic pv/inspection/severity:

{
  "panel_id": "panel_A",
  "pad_id": "pad_001",
  "robot_id": "robot_01",
  "model_version": "onnx_v1",
  "severity_score": 0.73,
  "status": "CRITICAL",
  "image_path": "captures/el_20260331_123456.jpg"
}

Expected result:

1. New row inserted into inspections
2. Alert evaluated and upserted by DB trigger logic
3. OPEN alert visible in Node-RED debug sidebar

## Full Flow Setup (Recommended)

Use this if you want one complete tab with realtime widgets and Supabase persistence:

1. Import `full_cder_supabase_dashboard_flow.json`
2. Open MQTT broker config node and set host/port/auth for your broker
3. Ensure Node-RED environment variables are set:
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY`
4. Deploy flow
5. Open dashboard and verify these widgets update:
  - Defect Severity gauge
  - Severity Over Time chart
  - Panel ID, Pad ID, Status, Robot, Model Version
  - DB Alert status
  - Open Alerts (DB)

Notes:

- The full flow is wired for topic `pv/inspection/severity`
- `status` defaults to `CRITICAL` when `severity_score >= 0.70` if sender omits status
- Keep `SUPABASE_SERVICE_ROLE_KEY` only in Node-RED environment variables (not inside Function code)
