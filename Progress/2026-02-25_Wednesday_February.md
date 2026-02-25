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

---

## Session Continuation – Portal web review fixes + date/range filter hardening

### Request
- Review web design/accessibility issues in portal UI and fix high/medium findings first.
- Fix portal date filtering issue seen in website usage (single date and range path causing failures/empty results).
- Provide session logs and commit guidance.

### Issues observed
- Accessibility review identified:
  - keyboard Enter hijack in review queue JS,
  - touch targets below recommended minimum,
  - ambiguous action label (`open`),
  - generic thumbnail alt text.
- Date filter implementation changes initially triggered SQLite translation error:
  - `The LINQ expression ... OccurredAtUtc >= __startUtc ... could not be translated`.
- Range filtering did not exist in UI/contract before this continuation.

### Changes implemented
- Accessibility and interaction fixes:
  - Removed Enter-key redirect override in `portal/wwwroot/js/site.js` so normal form submit behavior remains intact.
  - Increased compact action button size to 44px min touch target.
  - Increased queue-mini item touch target to 44px.
  - Replaced ambiguous `open` action text with `Inspect` and explicit ARIA label.
  - Replaced generic thumbnail alt text with contextual event-based alt text in dashboard/browser/detail/queue views.
- Date and range filtering support:
  - Added `DateFrom` and `DateTo` into request contract (`EventQueryRequest`) and view models.
  - Updated Dashboard and Event Browser filters to use `Date From` / `Date To`.
  - Preserved date-range parameters across pagination and CSV export links.
  - Implemented local-date range normalization in `EventQueryExtensions` with helper methods:
    - single-date compatibility (`Date`),
    - range parsing (`DateFrom`/`DateTo`),
    - local-day boundary to UTC conversion.
- SQLite-safe query path (critical fix for runtime error):
  - Kept SQL date filtering server-side for SQL Server.
  - For SQLite, applied date/range filtering in-memory after materializing filtered rows to avoid unsupported translation path.
  - Applied this provider-aware behavior in:
    - MVC controllers (`HomeController`, `EventsController`),
    - API controllers (`Api/DashboardController`, `Api/EventsController`),
    - CSV export path.
- Added portal regression coverage for date behavior:
  - New test case confirms both `date` and `dateFrom/dateTo` behavior for matching and non-matching days.

### Validation
- Successful checks during continuation:
  - `dotnet.exe build portal/Portal.Web.csproj -nologo` -> build succeeded earlier in session.
- Runtime bug signal used for correction:
  - User-provided portal exception trace confirmed SQLite translation failure in date filter LINQ.
- Pending/limited checks after final patch:
  - Some `dotnet` commands later failed in the current WSL environment with `UtilAcceptVsock:271: accept4 failed 110`, so final end-to-end rerun was not completed from this shell.
  - Final runtime verification expected from user run after restart.

### Plan Alignment
- Phase 7.3 (Portal Website MVP) remains in progress.
- This continuation improves production-readiness in Phase 7.3 by:
  - fixing reviewer UX/accessibility blockers,
  - enabling robust single-date and date-range filtering across MVC/API/export,
  - handling SQLite dev mode safely without breaking SQL Server behavior.

### File-level changes in this continuation

#### Added
- None.

#### Modified
- `Progress/2026-02-25_Wednesday_February.md`
- `portal/Contracts/EventQueryRequest.cs`
- `portal/Infrastructure/EventQueryExtensions.cs`
- `portal/Controllers/HomeController.cs`
- `portal/Controllers/EventsController.cs`
- `portal/Controllers/Api/DashboardController.cs`
- `portal/Controllers/Api/EventsController.cs`
- `portal/ViewModels/DashboardViewModel.cs`
- `portal/ViewModels/EventListPageViewModel.cs`
- `portal/Views/Home/Index.cshtml`
- `portal/Views/Events/Index.cshtml`
- `portal/Views/Events/Detail.cshtml`
- `portal/Views/Events/ReviewQueue.cshtml`
- `portal/wwwroot/css/site.css`
- `portal/wwwroot/js/site.js`
- `portal/tests/Portal.Web.Tests/PortalWorkflowTests.cs`

#### Deleted
- None.

## Session Continuation – Portal query audit

### Metadata
- Date: 2026-02-25
- Focus:
  - Audit the portal controllers and the query extension helpers for SQLite vs SQL Server performance traps.
  - Surface bottlenecks that make portal pages slow when the dev/test stack uses SQLite.
  - Keep the plan synced by referencing the most recent progress snapshot (`Progress/2026-02-25_Wednesday_February.md`).

### Previous Session Reference
- `Progress/2026-02-25_Wednesday_February.md` (latest entry at the top of the file).

### Summary
- Read the relevant controllers and query helpers to understand how data is filtered, paged, and aggregated for both SQL Server and SQLite.
- Identified the pushdown gaps where SQLite routes (`DbContext.Database.IsSqlServer()` false) fall back to pulling entire tables into memory before applying filters or pagination.
- Prepared an audit report (main response) that lists the top latency-impacting code paths and why they must be reworked for scale.

### Plan Alignment
- Phase 7.3 (Portal Website MVP) remains the focus; this audit highlights portal scalability risks before Phase 7.3 is considered production-ready.

### File-level changes
- None (read-only audit).

### Validation
- Not executed (analysis-only activity).

### Next Steps
- Consider pushing date-range filtering/pagination into SQLite queries (or switching to SQL Server) to avoid materializing entire datasets in the portal controllers.

---

## Session Continuation – SQLite pushdown + UI lag reduction

### Request
- Fix remaining lag during scrolling and page transitions after initial portal performance improvements.
- Keep logs updated and commit the changes.

### Changes implemented
- SQLite query-path performance (database pushdown):
  - Added SQLite date-range prefilter helper using SQL text-range on `occurred_at_utc` in `portal/Infrastructure/EventQueryExtensions.cs`.
  - Added provider-aware sort helpers to avoid unsupported SQLite `DateTimeOffset` `ORDER BY` translation.
  - Removed in-memory filtering/paging paths from MVC/API list/dashboard flows so count, page, and aggregate work happen in SQL.
- Review queue performance:
  - Kept `Take(7)` on query-side path for pending queue retrieval (no full pending materialization).
- UI/perceived performance reduction:
  - Event list and dashboard recent cards now prefer object thumbnails first (`thumb_path`) and only fall back to scene images.
  - Added thumbnail cache response header (`Cache-Control: private, max-age=300`).
  - Added image hints for list thumbnails (`decoding="async"`, `fetchpriority="low"`, fixed width/height).
  - Removed sticky table-header blur effect to reduce scroll paint/composite overhead.
  - Reduced default page size from 50 to 25 for lighter first paint and smoother table scrolling.

### Validation
- Build and tests:
  - `dotnet.exe build portal/Portal.Web.csproj -nologo` -> passed.
  - `dotnet.exe test portal/tests/Portal.Web.Tests/Portal.Web.Tests.csproj -nologo` -> passed (5/5).
- Synthetic benchmark (SQLite, 120,085 events):
  - Before:
    - events list path: ~279.31 ms
    - dashboard path: ~925.43 ms
    - review queue path: ~275.40 ms
  - After:
    - events list path: ~6.05 ms
    - dashboard path: ~49.92 ms
    - review queue path: ~0.09 ms

### Plan Alignment
- Phase 7.3 (Portal Website MVP) remains in progress.
- This continuation moves Phase 7.3 toward production-readiness by removing SQLite in-memory query bottlenecks and reducing front-end jank during high-frequency page interactions.

### File-level changes in this continuation

#### Added
- None.

#### Modified
- `portal/Infrastructure/EventQueryExtensions.cs`
- `portal/Controllers/HomeController.cs`
- `portal/Controllers/EventsController.cs`
- `portal/Controllers/Api/DashboardController.cs`
- `portal/Controllers/Api/EventsController.cs`
- `portal/Views/Events/Index.cshtml`
- `portal/Views/Home/Index.cshtml`
- `portal/wwwroot/css/site.css`
- `portal/appsettings.json`
- `Progress/2026-02-25_Wednesday_February.md`

#### Deleted
- None.
