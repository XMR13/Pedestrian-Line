# 2026-02-25 (Wednesday) – Production runbook for single-loop process

## Metadata
- Date: 2026-02-25
- Focus:
  - Deliver operator runbook for single-loop production mode (`main.py --portal-upload`).
  - Add reusable launch + service templates for auto-restart deployment.
  - Keep plan tracking synchronized with this operational milestone.

## Previous Session Reference
- Last logged session reviewed first: `Progress/2026-02-24_Tuesday_February.md`
- Carry-over from previous session:
  - Next step #1 was to add a production runbook for single-loop mode with service manager support.

## Summary
- Added a dedicated runbook for production operation in one process:
  - `docs/single_loop_production_runbook.md`
- Added a reusable launcher script with performance-oriented defaults:
  - `scripts/run_single_loop_live.sh`
- Added systemd templates for deployment + environment configuration:
  - `deploy/systemd/pedestrian-single-loop.service.example`
  - `deploy/systemd/pedestrian-single-loop.env.example`
- Updated top-level docs so operators can discover the runbook quickly from README.
- Updated `plan.md` to mark this runbook/operator-template checkpoint as completed under Phase 7 live mode.

## Plan Alignment
- `plan.md` Phase 7 (RTSP / Live Mode, in progress):
  - Added an implemented checkpoint for production single-loop operations:
    - runbook,
    - launcher script,
    - systemd template.
- Phase 7.1 (TensorRT backend) remains planned for full Jetson throughput optimization.
- Phase 7.2 and 7.3 status remain unchanged.

## File-level Changes

### Added
- `docs/single_loop_production_runbook.md`
- `scripts/run_single_loop_live.sh`
- `deploy/systemd/pedestrian-single-loop.service.example`
- `deploy/systemd/pedestrian-single-loop.env.example`
- `Progress/2026-02-25_Wednesday_February.md`

### Modified
- `README.md`
- `plan.md`

### Deleted
- None.

## Validation
- Static/sanity checks performed:
  - Reviewed CLI flags in `pedestrian_line_counter/main.py` to ensure runbook/script arguments are valid.
  - Script lint-level check to be done with `bash -n scripts/run_single_loop_live.sh`.
- Runtime validation not executed in this session:
  - Requires live RTSP source and portal endpoint availability on target environment.

## Next Steps
- Build and validate TensorRT `.engine` path on Jetson (Phase 7.1) and switch launcher defaults for production Jetson nodes.
- Add end-to-end smoke test script (process -> spool -> portal sync -> UI verification).
- Add optional health/status output (JSON file or endpoint) for monitoring integrations.

---

## Session Continuation – Minimal single-loop runtime tests + bounded smoke controls

### Request
- Add minimal tests to ensure single-loop service mode is runnable.
- Clarify whether service runs indefinitely.
- Ensure startup requires camera connectivity.

### Changes implemented
- Added new smoke tests:
  - `tests/test_main_single_loop_smoke.py`
    - `test_live_single_loop_integrated_upload_smoke`:
      - runs live-mode main loop with faked live source/stream reader and integrated uploader,
      - verifies spool run metadata (`run.json`) is produced and closed (`ended_at_utc`, `health_summary`),
      - verifies uploader path is invoked.
    - `test_live_startup_fails_when_camera_has_no_first_frame`:
      - verifies startup exits when first frame cannot be read from RTSP source.
- Updated launcher script to support controlled smoke duration without changing production default behavior:
  - `scripts/run_single_loop_live.sh`
  - New optional env vars:
    - `PLC_MAX_SECONDS`
    - `PLC_MAX_FRAMES`
  - If unset, runtime remains continuous.
- Updated runbook/templates to include optional bounded smoke controls:
  - `docs/single_loop_production_runbook.md`
  - `deploy/systemd/pedestrian-single-loop.env.example`

### Validation
- `python3 -m pytest -q tests/test_main_single_loop_smoke.py` -> `2 passed`.
- `python3 -m pytest -q tests/test_main_reconnect.py tests/test_main_single_loop_smoke.py` -> `4 passed`.
- `bash -n scripts/run_single_loop_live.sh` -> syntax OK.

### File-level changes in this continuation

#### Added
- `tests/test_main_single_loop_smoke.py`

#### Modified
- `scripts/run_single_loop_live.sh`
- `docs/single_loop_production_runbook.md`
- `deploy/systemd/pedestrian-single-loop.env.example`

#### Deleted
- None.
