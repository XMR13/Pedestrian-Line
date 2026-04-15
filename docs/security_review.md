# Security Review

This is the canonical security document for the current repo state.

It merges the previous threat-model and best-practices notes into one place so
the active FastAPI edge-service risks, assumptions, and recommended fixes stay
together.

## Executive Summary

The active production surface in this repo is the FastAPI-based edge service and
the local detector/spool runtime, not the older `portal/` application.

The main realistic risk for the current deployment shape is not a public-internet
attacker. It is an internal user, adjacent service, or compromised workstation
that can reach the Jetson-hosted FastAPI service over the LAN.

The most important findings are:

- unauthenticated read access to spool-backed data and service metadata,
- over-broad publication of spool contents through the `/evidence` static mount,
- storage and propagation of raw RTSP source values that may contain credentials,
- missing throttling on the shared local login,
- read-heavy endpoints that can put avoidable pressure on the Jetson.

## Scope And Assumptions

In scope:

- `pedestrian_line_counter/api.py`
- `pedestrian_line_counter/service.py`
- `pedestrian_line_counter/ui_auth.py`
- `pedestrian_line_counter/traffic_spool.py`
- `pedestrian_line_counter/event_uploader.py`
- `pedestrian_line_counter/event_contract.py`
- `pedestrian_line_counter/review_store.py`
- `scripts/run_edge_service.sh`
- `scripts/run_single_loop_live.sh`
- `docs/jetson_deployment_runbook.md`

Out of scope:

- the older `portal/` application as the primary production surface,
- public-internet reverse-proxy controls that are not enforced in repo code,
- local root compromise on the Jetson.

Assumptions used for ranking:

- the active deployment is the FastAPI edge service plus the detector/spool
  process,
- the service may be reachable over an internal LAN,
- users are few and effectively admin/operators,
- RTSP source strings may contain credentials if full URLs are used directly.

## System Model

Primary components:

- Detector/counting runtime
  - reads file or RTSP input and writes spool artifacts.
- Local spool store
  - holds `run.json`, `events.jsonl`, `status.json`, thumbnails, and reports.
- FastAPI edge service
  - exposes the operator UI, JSON endpoints, retention actions, and sync actions.
- Local review store
  - stores review decisions in SQLite under the spool root.
- Delivery worker
  - reads spool runs and sends payloads to the backend.

Trust boundaries:

- Operator browser -> FastAPI edge service
- Detector runtime -> local spool
- FastAPI edge service -> local spool
- Uploader -> backend API

## Top Risks

### SR-001: Unauthenticated read access to runtime and spool data

Affected behavior:

- several read routes expose configuration, recent runs, recent events, event
  details, and retention previews without the same protection used for the HTML
  UI.

Impact:

- internal users can enumerate evidence metadata, spool status, and service
  configuration without authenticating,
- the HTML login does not fully protect the underlying data surfaces.

Main repo areas:

- `pedestrian_line_counter/api.py`

Recommended fix:

- default all spool/data/config/status routes to protected access,
- split explicitly public routes from protected routes instead of protecting only
  selected UI pages.

### SR-002: Whole-spool publication through `/evidence`

Affected behavior:

- the service mounts the spool root as static content instead of exposing only
  intended image artifacts.

Impact:

- callers who know or guess spool-relative paths can access more than thumbnails:
  `run.json`, `events.jsonl`, `status.json`, `report.csv`, uploader state files,
  and the local review DB may all become reachable.

Main repo areas:

- `pedestrian_line_counter/api.py`
- `pedestrian_line_counter/_api_helpers.py`
- `pedestrian_line_counter/_api_common.py`

Recommended fix:

- replace the whole-root static mount with an authenticated file-serving route,
- restrict allowed files to evidence images under `thumbs/` and `scene/`.

### SR-003: RTSP credential leakage through stored source metadata

Affected behavior:

- raw RTSP source values are persisted in spool metadata and included in the
  outbound contract.

Impact:

- credential-bearing RTSP URLs can leak into run artifacts, local APIs, and
  backend payloads.

Main repo areas:

- `pedestrian_line_counter/main.py`
- `pedestrian_line_counter/traffic_spool.py`
- `pedestrian_line_counter/event_contract.py`

Recommended fix:

- store only a redacted or non-secret source identifier,
- remove raw `source_value` from payloads unless it is absolutely required.

### SR-004: Shared login without throttling

Affected behavior:

- the local UI login is a shared credential model without visible server-side
  throttling or lockout.

Impact:

- an internal attacker can brute-force the shared password if the service is
  exposed on the LAN.

Main repo areas:

- `pedestrian_line_counter/api.py`
- `pedestrian_line_counter/ui_auth.py`
- `pedestrian_line_counter/service.py`

Recommended fix:

- add lightweight login throttling,
- prefer individual internal auth when available.

### SR-005: Read-heavy endpoints can degrade Jetson availability

Affected behavior:

- several status/dashboard/event endpoints rescan spool files on demand.

Impact:

- repeated requests can create avoidable CPU and I/O pressure on the device.

Main repo areas:

- `pedestrian_line_counter/api.py`

Recommended fix:

- cache expensive read paths,
- precompute summaries where practical,
- apply basic request limiting if LAN exposure is necessary.

### SR-006: Coarse mutation API key model

Affected behavior:

- state-changing endpoints depend on a single local API key when enabled.

Impact:

- if the key leaks, retention and sync actions can be abused.

Main repo areas:

- `pedestrian_line_counter/api.py`
- `scripts/run_edge_service.sh`

Recommended fix:

- keep the service loopback-only where possible,
- rotate keys and limit who receives them,
- log all mutation-route usage.

## Detailed Findings

### High Severity

#### SBP-001

- Rule: `FASTAPI-AUTH-001`
- Theme: unauthenticated data exposure
- Main locations:
  - `pedestrian_line_counter/api.py`
- Key issue:
  - `config`, recent run/event endpoints, event detail payloads, and retention
    preview paths are not protected consistently.
- Fix:
  - require auth for operator-readable data routes, not only HTML pages.

#### SBP-002

- Rule: `FASTAPI-OPENAPI-001`
- Theme: over-broad static publication
- Main locations:
  - `pedestrian_line_counter/api.py`
  - `pedestrian_line_counter/_api_helpers.py`
  - `pedestrian_line_counter/_api_common.py`
- Key issue:
  - the entire spool root is mounted for static access.
- Fix:
  - expose only explicitly allowed evidence paths through a validated route.

#### SBP-003

- Rule: `FASTAPI-AUTH-002`
- Theme: source metadata leakage
- Main locations:
  - `pedestrian_line_counter/main.py`
  - `pedestrian_line_counter/traffic_spool.py`
  - `pedestrian_line_counter/event_contract.py`
- Key issue:
  - credential-bearing RTSP URLs may be stored and forwarded as normal metadata.
- Fix:
  - redact or replace source values before persistence and delivery.

### Medium Severity

#### SBP-004

- Rule: `FASTAPI-AUTH-001`
- Theme: brute-force resistance
- Main locations:
  - `pedestrian_line_counter/api.py`
  - `pedestrian_line_counter/ui_auth.py`
  - `pedestrian_line_counter/service.py`
- Key issue:
  - no visible login throttling or lockout for the shared local admin login.
- Fix:
  - add per-IP or per-username throttling and use a strong password.

#### SBP-005

- Rule: `FASTAPI-DEPLOY-001`
- Theme: Jetson-local availability pressure
- Main locations:
  - `pedestrian_line_counter/api.py`
- Key issue:
  - repeated spool rescans can be triggered through read-heavy endpoints.
- Fix:
  - add caching, summaries, or rate limiting.

### Low Severity / Operational Gaps

#### SBP-006

- Rule: `FASTAPI-AUTH-001`
- Theme: weak reviewer attribution
- Main locations:
  - `pedestrian_line_counter/review_store.py`
- Key issue:
  - review records store decision and notes, but not a meaningful reviewer identity.
- Fix:
  - add reviewer attribution when real auth is introduced.

## Recommended Fix Order

1. Replace the `/evidence` whole-spool mount with a restricted authenticated
   evidence-serving route.
2. Protect all runtime/config/spool-data routes, not just the HTML UI.
3. Stop persisting and forwarding raw `source_value` for RTSP inputs.
4. Add login throttling and basic request limiting for read-heavy routes.
5. Add reviewer attribution when the auth model is upgraded beyond the shared
   local admin login.

## Monitoring Notes

Until the code-level fixes are applied, the safest practical mitigations are:

- keep FastAPI on loopback where possible,
- limit LAN reachability to a very small admin/operator segment,
- avoid storing credential-bearing RTSP URLs directly in tracked configs,
- log accesses to config, event, and mutation endpoints.

## Current Status

This merged document replaces:

- `docs/security_best_practices_report.md`
- `docs/pedestrian_line_threat_model.md`

If future security work is done, update this file instead of splitting the same
deployment surface into multiple overlapping reports again.
