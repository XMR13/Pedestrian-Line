# Portal Architecture (Edge → Uploader → Website)

This document makes the portal/website integration concrete.

The repo you are in is the **edge pipeline** (Python). The portal website is a
separate service (ASP.NET Core + SQL Server) that ingests edge outputs via an
uploader.

## Goals (MVP)

- Show traffic crossing events per camera/day with thumbnails.
- Allow human review per event:
  - `Qualified`: `Yes` / `No` (default: pending)
  - `notes` (optional)
- Export reviewed labels for later training of an automatic classifier.

## Components

1. **Edge pipeline (this repo)**
- Input: MP4 or RTSP
- Output:
  - annotated video (optional)
  - spool run folder (recommended for portal ingestion)

2. **Spool (filesystem-first)**
- Written by the edge pipeline.
- Contents per run:
  - `run.json`
  - `events.jsonl`
  - `report.csv` (human-readable)
  - `thumbs/<event_uid>.jpg` (optional)

3. **Uploader (edge-side async process)**
- Reads spool folders and pushes:
  - run metadata
  - event records
  - thumbnails
- Must be idempotent (safe to retry).

4. **Portal API + Website**
- ASP.NET Core API with SQL Server for metadata.
- Thumbnails stored on portal disk (evidence folder).
- Website UI for dashboards and human review.

## Data Flow (Sequence)

```mermaid
sequenceDiagram
  participant Edge as Edge Pipeline (Python)
  participant Spool as Spool Folder
  participant Up as Uploader (Python)
  participant API as Portal API (ASP.NET)
  participant DB as SQL Server
  participant FS as Portal Disk (Evidence)

  Edge->>Spool: write run.json/events.jsonl/thumbs/report.csv
  Up->>Spool: scan for pending runs
  Up->>API: upsert run (run_uid)
  API->>DB: insert/update runs
  Up->>API: upsert events batch (event_uid)
  API->>DB: insert/update events
  Up->>API: upload thumbnails (event_uid.jpg)
  API->>FS: store evidence file
  API->>DB: update events.thumb_path
```

## Idempotency Rules

- `run_uid` uniquely identifies a run. Upsert is keyed by `run_uid`.
- `event_uid` uniquely identifies an event. Upsert is keyed by `event_uid`.
- Thumbnail upload is keyed by `event_uid`:
  - safe behavior: if file exists and size matches, return 200/204 without rewriting
  - if missing, write file then update DB

## Portal DB Schema (MVP)

### `runs`

- `run_uid` (PK)
- `site_id`, `camera_id`
- `started_at_utc`, `ended_at_utc` (optional)
- `source_type`, `source_value`
- `model_version`, `cfg_version`
- `line_mode`, `line_id`
- `fps`, `frame_width`, `frame_height`
- `health_summary_json` (JSON)
- `report_csv_relpath` (optional)

### `events`

- `event_uid` (PK)
- `run_uid` (FK)
- `site_id`, `camera_id`
- `occurred_at_utc`
- `frame_index`, `video_time_s`
- `direction` (`A_TO_B` / `B_TO_A`)
- `track_id`
- `class_id`, `class_name`, `confidence`
- `bbox_json` (int[4] as JSON)
- `thumb_path` (portal disk path, optional)

### `event_reviews`

- `event_uid` (PK, FK -> events)
- `review_status` (`PENDING`, `QUALIFIED`, `NOT_QUALIFIED`)
- `reviewed_at_utc` (nullable)
- `reviewed_by` (nullable)
- `notes` (nullable)

### `camera_criteria`

This is where you store what "Qualified" means for each camera.

- `site_id`, `camera_id` (composite PK)
- `criteria_title`
- `criteria_description` (text/markdown)

## Portal API Endpoints (MVP Contract)

Authentication (MVP):

- `X-API-Key: <secret>` header for uploader requests.

Endpoints:

- `POST /api/runs/upsert`
  - Upsert by `run_uid`.
- `POST /api/events/upsert`
  - Batch upsert by `event_uid`.
- `POST /api/events/{event_uid}/thumbnail`
  - `multipart/form-data` upload.
- `POST /api/events/{event_uid}/review`
  - Set `Qualified Yes/No` + notes.
- `GET /api/events`
  - Filters: site/camera/date/direction/class/review status.
- `GET /api/events/{event_uid}/thumbnail`
  - Serve image for UI.
- `GET /api/dashboard/summary`
  - Aggregates counts and review stats.

### Example Payloads (Contract v1)

`POST /api/runs/upsert`

```json
{
  "contract_version": "v1",
  "run_uid": "7c6b4e9f4e1a4f6bbbf6e88f4c62f7eb",
  "site_id": "subang",
  "camera_id": "cam_01",
  "started_at_utc": "2026-02-20T01:00:00Z",
  "ended_at_utc": "2026-02-20T01:15:31Z",
  "source_type": "rtsp",
  "source_value": "rtsp://example/cam01",
  "model_version": "vehicle_subclasses.onnx",
  "cfg_version": "default",
  "line_mode": "single",
  "line_id": "line_subang_cam01",
  "fps": 30.0,
  "frame_width": 1920,
  "frame_height": 1080,
  "health_summary_json": {
    "frames_total": 27158,
    "effective_fps": 29.3
  },
  "report_csv_relpath": "report.csv"
}
```

`POST /api/events/upsert`

```json
{
  "contract_version": "v1",
  "events": [
    {
      "contract_version": "v1",
      "event_uid": "b9e0f6f7f9fd4b0aa0ea7fbe36c4c2a0",
      "run_uid": "7c6b4e9f4e1a4f6bbbf6e88f4c62f7eb",
      "site_id": "subang",
      "camera_id": "cam_01",
      "occurred_at_utc": "2026-02-20T01:02:14Z",
      "frame_index": 4050,
      "video_time_s": 135.0,
      "direction": "A_TO_B",
      "track_id": 148,
      "class_id": 2,
      "class_name": "truck",
      "confidence": 0.91,
      "bbox_xyxy": [561, 337, 731, 569],
      "line_mode": "single",
      "occurred_at_utc_source": "wall_clock",
      "thumb_relpath": "thumbs/b9e0f6f7f9fd4b0aa0ea7fbe36c4c2a0.jpg",
      "scene_relpath": "scene/b9e0f6f7f9fd4b0aa0ea7fbe36c4c2a0.jpg"
    }
  ]
}
```

## Uploader Runtime State

The edge uploader stores per-run progress in:

- `<run_dir>/.portal_upload_state.json`

This allows safe restart/resume and keeps network retries independent from the
inference process. Contract idempotency remains the source of truth.

## UI Pages (MVP)

- Dashboard (summary + filters)
- Runs list
- Event browser (table/grid)
- Review queue (fast Yes/No + notes)
- Export reviewed labels (CSV)

## Notes About “Specific Vehicle Characteristics”

In v1, the edge model detects vehicle subclasses (truck/tronton/etc).

The "specific characteristics" (e.g. carrying waste paper vs bricks) are handled
by:

- human review in portal: `Qualified Yes/No` (+ notes)
- per-camera criteria text in `camera_criteria`

Later, those human labels can train a dedicated classifier so the portal can
auto-suggest `Qualified` while keeping human review as source of truth.
