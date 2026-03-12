# R&R Study and Remote Optometrist Dashboard

## R&R (Repeatability and Reproducibility) Study

The system records session metadata (in `logs/combined_metadata.csv` and per-session JSON) that supports:

- **Repeatability**: Same operator, same subject/phoropter, multiple tests — you want stable outcomes (e.g. similar final prescription, completion path, duration).
- **Reproducibility**: Different operators, sites, or phoropters — you want consistent behavior and outcomes across the system.

### Metadata used for R&R

- **Identifiers**: `Session_ID`, `Phoropter_ID`, `Operator_Name`, `Customer_Name` (and age/gender).
- **Outcomes**: `Completion_Status`, final prescription (`Final_R_*`, `Final_L_*`), AR and lensometry if provided.
- **Process quality**: `Manual_Count`, `QnA_Count`, `Phase_Jump_Count`, `Unable_To_Read_Count`, `Total_Interactions`, `Duration_Seconds`, `Phases_Completed`.

### Metrics available on the dashboard

- **By operator / by phoropter**: Session count, mean duration (seconds), completion rate, mean phase-jump rate (phase jumps relative to total interactions).
- **Filters**: Date range (`from`, `to`), operator, phoropter, completion status — apply these to focus on a subset for repeatability or reproducibility analysis.

### Using the dashboard for R&R

- **Repeatability**: Filter by one operator and one phoropter (and optionally date range). Use "By operator" / "By phoropter" and the recent sessions list to compare runs. Export CSV for offline analysis (e.g. variance of spherical equivalent, completion rate).
- **Reproducibility**: Compare "By operator" and "By phoropter" across different operators and devices; use filters to compare time windows (e.g. by day or week).

### Optional: subject-level repeatability

For strict repeatability by *subject*, a stable subject/patient ID is useful. The system currently has `Customer_Name` in metadata; if R&R subjects are tracked by name, that can suffice. Otherwise, an optional `subject_id` (or `customer_id`) can be added to session start and to metadata later for grouping multiple tests per subject.

---

## Remote Optometrist Dashboard

### Access

- **URL**: `/dashboard` (e.g. http://localhost:5050/dashboard when the API server is running).
- **Auth**: If the server has `DASHBOARD_SECRET` or `DASHBOARD_PASSWORD` set, the dashboard and all `/api/dashboard/*` endpoints require authentication:
  - HTTP Basic: any username, password = the secret.
  - Or header: `Authorization: Bearer <secret>`.

### Data source: Local logs vs Supabase

In **Data source & filters** you can switch between:

- **Local logs** — reads from the server’s `logs/combined_metadata.csv` (and sessions written locally). Use for local testing.
- **Supabase** — fetches session metadata from the configured Supabase Storage bucket (same bucket used when `REMOTE_STORAGE=supabase`). Use to view all sessions stored in the cloud.

Supabase must be configured (`REMOTE_STORAGE=supabase`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and optionally `SUPABASE_BUCKET`). If you choose Supabase and it is not configured or fails, the dashboard shows an error and empty data.

### What you can do

1. **View**
   - Total sessions (filtered by date/operator/phoropter), active sessions count, counts by day, and a recent sessions table.
   - R&R section: tables by operator and by phoropter (count, mean duration, completion rate, phase jump rate).

2. **Control**
   - **New tests enabled**: Global on/off for starting new sessions. When off, `POST /api/session/start` returns 403 with `code: "tests_disabled"`.
   - **Daily limit**: Optional cap (number) with scope "Global" or "Per phoropter". When the count of sessions started today (from metadata + in-memory) reaches the limit, new starts return 403 with `code: "daily_limit_reached"`.
   - **Per-phoropter enable**: Enable/disable starting new tests per phoropter ID. Disabled phoropters get 403 with `code: "phoropter_disabled"`.

3. **Export**
   - "Download CSV" uses the current filters and returns the same columns as `combined_metadata.csv` for offline analysis.

### API (for integrations)

- `GET /api/dashboard/config` — current config (tests_enabled, daily_limit, daily_limit_scope, per_phoropter_enabled).
- `PUT /api/dashboard/config` — update config (JSON body).
- `GET /api/dashboard/stats?from=&to=&operator=&phoropter=&source=local|supabase` — total, by_day, recent_sessions, active_sessions_count. `source=supabase` fetches from Supabase Storage.
- `GET /api/dashboard/rr?from=&to=&operator=&phoropter=&source=local|supabase` — by_operator and by_phoropter aggregates.
- `GET /api/dashboard/export?from=&to=&operator=&phoropter=&source=local|supabase` — CSV download.

Config is stored in `logs/dashboard_config.json` and is read on each request so changes take effect immediately.
