# Security Best Practices Report

## Executive Summary

The active FastAPI edge service has two material security gaps for an internal-only deployment: it exposes spool-backed evidence/data over unauthenticated read routes, and it mounts the entire spool root as static content instead of limiting exposure to image artifacts. In this repo's current deployment model, the main realistic attacker is not the public internet but an internal user, compromised workstation, or adjacent internal service that can reach the Jetson-hosted FastAPI service over the LAN.

## Critical / High Severity

### SBP-001

- Rule ID: `FASTAPI-AUTH-001`
- Severity: High
- Location:
  - `pedestrian_line_counter/api.py:1291`
  - `pedestrian_line_counter/api.py:1295`
  - `pedestrian_line_counter/api.py:1307`
  - `pedestrian_line_counter/api.py:1333`
  - `pedestrian_line_counter/api.py:1558`
- Evidence:
  - `config_view()` has no auth dependency at `pedestrian_line_counter/api.py:1291-1293`.
  - `recent_runs()` has no auth dependency at `pedestrian_line_counter/api.py:1295-1305`.
  - `recent_events()` has no auth dependency at `pedestrian_line_counter/api.py:1307-1331`.
  - `event_detail_payload()` has no auth dependency at `pedestrian_line_counter/api.py:1333-1341`.
  - `retention_preview()` has no auth dependency at `pedestrian_line_counter/api.py:1558-1560`.
- Impact:
  - Any reachable internal client can enumerate runs, events, evidence URLs, review state, service configuration, spool paths, and uploader status without authenticating.
  - This weakens the intended "users == admin" boundary because the HTML login protects the UI pages, but not the JSON data surfaces that back them.
- Fix:
  - Require UI auth for all operator-readable routes that expose spool or service state, not just HTML pages.
  - Split routes into explicit public vs protected routers and default all runtime/introspection/data routes to protected.
- Mitigation:
  - Keep the service on loopback only until route auth is enforced consistently.
  - Restrict access at the network layer to a small operator/admin subnet if LAN exposure is unavoidable.
- False positive notes:
  - If another authenticated reverse proxy is always in front of the service, severity drops, but no such control is visible in this repo.

### SBP-002

- Rule ID: `FASTAPI-OPENAPI-001`
- Severity: High
- Location:
  - `pedestrian_line_counter/api.py:1164`
  - `pedestrian_line_counter/_api_helpers.py:126-146`
  - `pedestrian_line_counter/_api_helpers.py:865-866`
  - `pedestrian_line_counter/_api_helpers.py:566`
  - `pedestrian_line_counter/_api_common.py:17`
- Evidence:
  - The service mounts the entire spool root directly: `app.mount("/evidence", StaticFiles(directory=str(spool_dir), check_dir=False), name="evidence")` at `pedestrian_line_counter/api.py:1164`.
  - Event summaries carry spool-relative file paths and convert them into public URLs at `pedestrian_line_counter/_api_helpers.py:122-146` and `pedestrian_line_counter/_api_helpers.py:865-866`.
  - The review DB filename is stored inside the spool root as `.edge_ui_reviews.sqlite3` at `pedestrian_line_counter/_api_common.py:17`.
- Impact:
  - A requester who knows or guesses spool-relative paths can fetch more than thumbnails: `run.json`, `events.jsonl`, `status.json`, `report.csv`, uploader state files, and the review SQLite DB may all become web-accessible.
  - This creates direct confidentiality risk for operational data, review notes, file paths, and possibly credentials embedded in spool metadata.
- Fix:
  - Stop mounting the whole spool root as static content.
  - Serve only explicitly allowed evidence subpaths, ideally through a route that validates auth and restricts file types to image artifacts under `thumbs/` and `scene/`.
- Mitigation:
  - If an immediate code change is not possible, place the service behind a network control that only operator/admin machines can reach and remove any path disclosure from unauthenticated routes.
- False positive notes:
  - Starlette `StaticFiles` does protect against basic traversal, so this is not a traversal finding; it is an over-broad publication finding.

### SBP-003

- Rule ID: `FASTAPI-AUTH-002`
- Severity: High
- Location:
  - `pedestrian_line_counter/main.py:1638`
  - `pedestrian_line_counter/traffic_spool.py:117-126`
  - `pedestrian_line_counter/event_contract.py:65-90`
  - `docs/jetson_dual_service_runbook.md:117-143`
- Evidence:
  - The spool writer stores raw source details in run metadata, including RTSP sources: `source = {"type": "rtsp", "value": str(rtsp_url)}` at `pedestrian_line_counter/main.py:1638`.
  - `run.json` persists `source` and `source_value` at `pedestrian_line_counter/traffic_spool.py:117-126`.
  - The outbound delivery contract forwards `source_value` in the run payload at `pedestrian_line_counter/event_contract.py:65-90`.
  - The deployment docs show credential-bearing RTSP URLs as a normal configuration shape at `docs/jetson_dual_service_runbook.md:117-143`.
- Impact:
  - If RTSP URLs contain camera credentials, those credentials are written to disk, exposed to the FastAPI service, and propagated to the backend payload contract.
  - Combined with `SBP-001` and `SBP-002`, this can become direct credential disclosure.
- Fix:
  - Do not store raw credential-bearing RTSP URLs in spool artifacts.
  - Persist only a redacted camera/source identifier or a sanitized URI with credentials removed.
  - Remove `source_value` from outbound payloads unless there is a hard business requirement.
- Mitigation:
  - Use env indirection for RTSP URLs and store only the env key or camera name in spool metadata.
- False positive notes:
  - If production RTSP URLs never include embedded credentials, the impact is lower, but the current documented configuration strongly suggests they can.

## Medium Severity

### SBP-004

- Rule ID: `FASTAPI-AUTH-001`
- Severity: Medium
- Location:
  - `pedestrian_line_counter/api.py:1182-1203`
  - `pedestrian_line_counter/api.py:1211-1245`
  - `pedestrian_line_counter/ui_auth.py:22-57`
  - `pedestrian_line_counter/service.py:177-195`
- Evidence:
  - Login accepts a single configured username/password and issues a signed cookie, but no visible rate limit, attempt counter, or lockout is enforced.
  - The login handlers compare credentials directly and return immediately on failure at `pedestrian_line_counter/api.py:1190-1191` and `pedestrian_line_counter/api.py:1226-1230`.
- Impact:
  - On a LAN-exposed service, an internal attacker or compromised workstation can brute-force the shared admin credential without server-side throttling.
  - Because the model is effectively a shared admin account, a single guessed password grants broad operator access.
- Fix:
  - Add lightweight login throttling per source IP and per username.
  - Prefer integration with an internal auth provider when available so access is attributable to individual users rather than a shared account.
- Mitigation:
  - Keep the service loopback-only where possible, and restrict LAN exposure to a small operator/admin segment.
  - Use a long, random password for `EDGE_UI_PASSWORD`.
- False positive notes:
  - If access is already tightly restricted to a jump host or bastion, the practical likelihood drops.

### SBP-005

- Rule ID: `FASTAPI-DEPLOY-001`
- Severity: Medium
- Location:
  - `pedestrian_line_counter/api.py:131-248`
  - `pedestrian_line_counter/api.py:330-840`
- Evidence:
  - `status_payload()`, `metrics_payload()`, `list_recent_runs()`, `list_recent_events()`, and `dashboard_payload()` walk the filesystem spool and parse all runs/events on each request.
  - Examples:
    - `_iter_all_runs()` at `pedestrian_line_counter/api.py:810-821`
    - `_iter_all_events()` at `pedestrian_line_counter/api.py:823-840`
    - dashboard aggregation at `pedestrian_line_counter/api.py:673-708`
- Impact:
  - An internal client can cause avoidable CPU/IO pressure on the Jetson by repeatedly hitting read-heavy endpoints.
  - This is an availability risk rather than a privilege-escalation risk, but it matters because the edge device is compute-constrained.
- Fix:
  - Add rate limiting or caching for expensive read endpoints.
  - Consider precomputed summaries for dashboard/status views instead of rescanning the full spool on each request.
- Mitigation:
  - Put the service behind a small internal reverse proxy with request limiting if the app must be LAN-exposed.
- False positive notes:
  - With very small spool sizes and very few users, the current implementation may perform adequately, but the repo does not enforce this operational assumption.

## Low Severity / Operational Gaps

### SBP-006

- Rule ID: `FASTAPI-AUTH-001`
- Severity: Low
- Location:
  - `pedestrian_line_counter/review_store.py:14-37`
  - `pedestrian_line_counter/review_store.py:73-137`
- Evidence:
  - Review records store decision and notes, but not the authenticated actor identity.
- Impact:
  - In a shared-admin deployment, you lose per-user accountability for review actions.
- Fix:
  - When real auth is introduced, store reviewer identity alongside the review record.
- Mitigation:
  - Treat the current review DB as an MVP operational store, not a strong audit trail.
- False positive notes:
  - Because you stated that users are effectively admins only, this is lower priority than the exposure issues above.

## Recommended Fix Order

1. Replace the `/evidence` whole-spool static mount with a restricted, authenticated evidence-serving route.
2. Enforce auth on all spool/data/config/status endpoints, not only on HTML pages.
3. Stop persisting and forwarding raw `source_value` for RTSP inputs; redact or replace it with a non-secret source identifier.
4. Add basic rate limiting for login and expensive read endpoints.
5. Later, replace the shared local admin login with per-user internal authentication and reviewer attribution.

