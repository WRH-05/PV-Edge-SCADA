# Supabase Database Setup

This folder contains the SQL needed for a production-ready baseline for:
- panel/pad asset tracking
- inspection history
- threshold management
- automatic alert lifecycle
- dashboard views

## Files

- `000_full_migration.sql` — Single-file migration that creates schema, functions, triggers, and views in one run.

## How To Apply In Supabase

Run this single file in the Supabase SQL Editor:

1. `000_full_migration.sql`

The migration creates all tables, indexes, functions, triggers, and views in one atomic execution.

## Data Flow Expected

1. Raspberry Pi publishes MQTT payloads.
2. Node-RED receives payloads and inserts into `inspections`.
3. DB trigger runs automatically after each insert and updates `alerts`.
4. Node-RED dashboard reads from:
   - `vw_latest_pad_status`
   - `vw_pad_weekly_trend`
   - `vw_open_alerts`

## Insert Contract For `inspections`

At minimum, Node-RED should send:
- `captured_at` (ISO timestamp)
- `robot_id`
- `panel_id`
- `pad_id`
- `severity_score` (0..1)
- `status` (for example: `OK`, `WARN`, `CRITICAL`)

Recommended extras:
- `image_path`
- `model_version`
- `raw_payload` (full JSON payload from the Pi)

## Acknowledge Alerts (Optional)

When an engineer acknowledges an active alert:

```sql
update alerts
set
  state = 'ACKNOWLEDGED',
  acknowledged_by = 'engineer_name',
  acknowledged_at = now(),
  updated_at = now()
where id = :alert_id
  and active = true;
```

## Notes

- Keep Supabase service role key only in Node-RED server environment variables.
- Keep Raspberry Pi focused on inference and MQTT publishing.
- Keep the existing CSV on Pi as local fallback/audit backup.
