# MVP UI Runbook

This runbook explains how to start the new FastAPI-served MVP UI, where to open it,
and how to capture logs for later debugging.

## Prerequisites

- Project dependencies installed:

```bash
uv sync
```

- A spool directory that contains run folders and event evidence.
  Example:

```text
data/traffic_runs
```

If the spool is empty, the UI still starts, but dashboard/review data will be sparse.

## Start The UI Service

Run from the repo root:

```bash
uv run python -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 127.0.0.1 \
  --port 8080
```

Open:

- Login: `http://127.0.0.1:8080/ui/login`
- Dashboard: `http://127.0.0.1:8080/ui/dashboard`
- Review Queue: `http://127.0.0.1:8080/ui/review`

The service also exposes JSON endpoints on the same port, including:

- `/healthz`
- `/status`
- `/metrics`
- `/events/recent`
- `/review/queue`

## Start With Logs

Create a local logs directory and write stdout/stderr into a timestamped file:

```bash
mkdir -p logs
uv run python -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 127.0.0.1 \
  --port 8080 \
  2>&1 | tee "logs/mvp-ui-$(date +%Y%m%d-%H%M%S).log"
```

This keeps logs visible in the terminal while also saving them to disk.

## Stop The Service

In the same terminal:

```bash
Ctrl+C
```

## Useful Variants

Enable login gating for the UI:

```bash
export EDGE_UI_USERNAME="admin"
export EDGE_UI_PASSWORD="change-me"
uv run python -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 127.0.0.1 \
  --port 8080
```

When `EDGE_UI_PASSWORD` is set:

- `/` redirects to `/ui/login`
- UI pages require a local cookie session
- review actions also require the same login session

If `EDGE_UI_PASSWORD` is not set, login gating is disabled and the UI opens directly.

Use a different spool directory:

```bash
uv run python -m pedestrian_line_counter.service \
  --spool-dir /absolute/path/to/spool \
  --host 127.0.0.1 \
  --port 8080
```

Protect state-changing endpoints with a local API key:

```bash
export EDGE_SERVICE_API_KEY="change-me"
uv run python -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 127.0.0.1 \
  --port 8080
```

## Runtime Notes

- Review decisions are stored locally in SQLite under the spool root as:

```text
<spool-dir>/.edge_ui_reviews.sqlite3
```

- Evidence images are served directly from the spool through `/evidence/...`.
- The dashboard chart is currently placeholder-first; totals and review state are live.
