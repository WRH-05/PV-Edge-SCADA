-- 000_full_migration.sql
-- Ready-to-run full Supabase migration for EL degradation tracking.

begin;

create table if not exists assets (
  id bigint generated always as identity primary key,
  panel_id text not null,
  pad_id text not null,
  location text,
  install_date date,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (panel_id, pad_id)
);

create table if not exists thresholds (
  id bigint generated always as identity primary key,
  panel_id text,
  pad_id text,
  warn_threshold numeric(6,4) not null default 0.55,
  critical_threshold numeric(6,4) not null default 0.70,
  recovery_threshold numeric(6,4) not null default 0.60,
  slope_threshold numeric(12,10) not null default 0.0000015,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (warn_threshold < critical_threshold),
  check (recovery_threshold < critical_threshold)
);

create table if not exists inspections (
  id bigint generated always as identity primary key,
  captured_at timestamptz not null,
  robot_id text not null,
  panel_id text not null,
  pad_id text not null,
  severity_score numeric(6,4) not null,
  status text not null,
  image_path text,
  model_version text,
  raw_payload jsonb not null default '{}'::jsonb,
  inserted_at timestamptz not null default now(),
  check (severity_score >= 0 and severity_score <= 1)
);

create index if not exists inspections_panel_pad_time_idx
  on inspections (panel_id, pad_id, captured_at desc);

create index if not exists inspections_inserted_at_idx
  on inspections (inserted_at desc);

create table if not exists alerts (
  id bigint generated always as identity primary key,
  panel_id text not null,
  pad_id text not null,
  state text not null check (state in ('OPEN', 'ACKNOWLEDGED', 'RESOLVED')),
  reason text not null,
  opened_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  resolved_at timestamptz,
  last_severity numeric(6,4),
  slope_4w numeric(14,10),
  active boolean not null default true,
  acknowledged_by text,
  acknowledged_at timestamptz
);

create unique index if not exists alerts_one_active_per_pad_idx
  on alerts (panel_id, pad_id)
  where active = true;

create index if not exists alerts_active_idx
  on alerts (active, panel_id, pad_id, updated_at desc);

create table if not exists maintenance_events (
  id bigint generated always as identity primary key,
  panel_id text not null,
  pad_id text not null,
  replaced_at timestamptz not null,
  engineer text,
  notes text,
  created_at timestamptz not null default now()
);

insert into thresholds (
  panel_id,
  pad_id,
  warn_threshold,
  critical_threshold,
  recovery_threshold,
  slope_threshold
)
select null, null, 0.55, 0.70, 0.60, 0.0000015
where not exists (
  select 1
  from thresholds t
  where t.panel_id is null and t.pad_id is null
);

create or replace function get_effective_threshold(
  p_panel_id text,
  p_pad_id text
)
returns table (
  warn_threshold numeric,
  critical_threshold numeric,
  recovery_threshold numeric,
  slope_threshold numeric
)
language sql
stable
as $$
  select
    t.warn_threshold,
    t.critical_threshold,
    t.recovery_threshold,
    t.slope_threshold
  from thresholds t
  where
    (t.panel_id = p_panel_id and t.pad_id = p_pad_id)
    or (t.panel_id is null and t.pad_id is null)
  order by
    case when t.panel_id is null and t.pad_id is null then 1 else 0 end,
    t.updated_at desc
  limit 1;
$$;

create or replace function evaluate_pad_alert(
  p_panel_id text,
  p_pad_id text
)
returns table (
  state text,
  reason text,
  current_score numeric,
  slope_4w numeric
)
language plpgsql
as $$
declare
  v_warn numeric;
  v_critical numeric;
  v_recovery numeric;
  v_slope_thr numeric;
  v_current numeric;
  v_slope double precision;
begin
  select
    th.warn_threshold,
    th.critical_threshold,
    th.recovery_threshold,
    th.slope_threshold
  into
    v_warn,
    v_critical,
    v_recovery,
    v_slope_thr
  from get_effective_threshold(p_panel_id, p_pad_id) th;

  select i.severity_score
  into v_current
  from inspections i
  where i.panel_id = p_panel_id
    and i.pad_id = p_pad_id
  order by i.captured_at desc
  limit 1;

  select regr_slope(
    i.severity_score::double precision,
    extract(epoch from i.captured_at)
  )
  into v_slope
  from inspections i
  where i.panel_id = p_panel_id
    and i.pad_id = p_pad_id
    and i.captured_at >= now() - interval '28 days';

  if v_current is null then
    return;
  end if;

  if v_current >= v_critical then
    return query select 'OPEN'::text, 'CRITICAL_THRESHOLD'::text, v_current, coalesce(v_slope, 0)::numeric;
    return;
  end if;

  if coalesce(v_slope, 0) >= coalesce(v_slope_thr, 0) then
    return query select 'OPEN'::text, 'RISING_TREND'::text, v_current, coalesce(v_slope, 0)::numeric;
    return;
  end if;

  if v_current <= v_recovery then
    return query select 'RESOLVED'::text, 'RECOVERY_THRESHOLD'::text, v_current, coalesce(v_slope, 0)::numeric;
    return;
  end if;

  if v_current >= v_warn then
    return query select 'OPEN'::text, 'WARN_THRESHOLD'::text, v_current, coalesce(v_slope, 0)::numeric;
  else
    return query select 'RESOLVED'::text, 'NORMAL_RANGE'::text, v_current, coalesce(v_slope, 0)::numeric;
  end if;
end;
$$;

create or replace function upsert_alert_from_latest_inspection(
  p_panel_id text,
  p_pad_id text
)
returns void
language plpgsql
as $$
declare
  v_state text;
  v_reason text;
  v_current numeric;
  v_slope numeric;
  v_active_id bigint;
begin
  select e.state, e.reason, e.current_score, e.slope_4w
  into v_state, v_reason, v_current, v_slope
  from evaluate_pad_alert(p_panel_id, p_pad_id) e
  limit 1;

  if v_state is null then
    return;
  end if;

  select a.id
  into v_active_id
  from alerts a
  where a.panel_id = p_panel_id
    and a.pad_id = p_pad_id
    and a.active = true
  order by a.opened_at desc
  limit 1;

  if v_state = 'OPEN' then
    if v_active_id is null then
      insert into alerts (
        panel_id,
        pad_id,
        state,
        reason,
        opened_at,
        updated_at,
        last_severity,
        slope_4w,
        active
      )
      values (
        p_panel_id,
        p_pad_id,
        'OPEN',
        v_reason,
        now(),
        now(),
        v_current,
        v_slope,
        true
      );
    else
      update alerts
      set
        state = case when state = 'ACKNOWLEDGED' then 'ACKNOWLEDGED' else 'OPEN' end,
        reason = v_reason,
        updated_at = now(),
        last_severity = v_current,
        slope_4w = v_slope
      where id = v_active_id;
    end if;
  else
    if v_active_id is not null then
      update alerts
      set
        state = 'RESOLVED',
        reason = v_reason,
        updated_at = now(),
        resolved_at = now(),
        active = false,
        last_severity = v_current,
        slope_4w = v_slope
      where id = v_active_id;
    end if;
  end if;
end;
$$;

create or replace function trg_inspections_apply_alert()
returns trigger
language plpgsql
as $$
begin
  perform upsert_alert_from_latest_inspection(new.panel_id, new.pad_id);
  return new;
end;
$$;

drop trigger if exists inspections_apply_alert_trigger on inspections;

create trigger inspections_apply_alert_trigger
after insert on inspections
for each row
execute function trg_inspections_apply_alert();

create or replace view vw_latest_pad_status as
select distinct on (i.panel_id, i.pad_id)
  i.panel_id,
  i.pad_id,
  i.captured_at,
  i.severity_score,
  i.status,
  i.robot_id,
  i.model_version,
  i.image_path
from inspections i
order by i.panel_id, i.pad_id, i.captured_at desc;

create or replace view vw_pad_weekly_trend as
with weekly as (
  select
    i.panel_id,
    i.pad_id,
    date_trunc('week', i.captured_at) as week_start,
    avg(i.severity_score)::numeric(6,4) as avg_score,
    max(i.severity_score)::numeric(6,4) as max_score,
    min(i.severity_score)::numeric(6,4) as min_score,
    count(*)::int as samples
  from inspections i
  where i.captured_at >= now() - interval '12 weeks'
  group by i.panel_id, i.pad_id, date_trunc('week', i.captured_at)
)
select *
from weekly
order by panel_id, pad_id, week_start desc;

create or replace view vw_open_alerts as
select
  a.id,
  a.panel_id,
  a.pad_id,
  a.state,
  a.reason,
  a.opened_at,
  a.updated_at,
  a.last_severity,
  a.slope_4w,
  a.acknowledged_by,
  a.acknowledged_at
from alerts a
where a.active = true
order by a.updated_at desc;

commit;
