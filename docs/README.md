# Documentation Guide

This folder is the main documentation home for the project.

Use this file to decide which document to read first instead of guessing from
older filenames.

## Read First

- `jetson_deployment_runbook.md`
  - Main deployment guide for the active edge runtime:
    `single_loop.service` + `edge_service.service`.
- `ezviz_windows_bridge_rtsp_runbook.md`
  - Current MVP guide for getting the Windows EZVIZ bridge to publish RTSP to
    the AI box.
- `line_direction.md`
  - Explains the counting geometry and direction logic.

## Deployment And Operations

- `jetson_deployment_runbook.md`
  - Jetson setup, env files, services, spool sharing, manual UI bring-up,
    validation, and troubleshooting.
- `remote_ai_box_public_hosting_guide.md`
  - Reverse-proxy and remote access guidance when the AI box needs to be
    reachable from outside the site LAN.
- `tensorrt_engine_bringup.md`
  - TensorRT bring-up notes, compatibility pitfalls, and the validated working
    operating mode on Jetson.

## Integration And Architecture

- `portal_architecture.md`
  - Canonical reference for the legacy ASP.NET portal surface, including
    architecture, local setup, uploader integration, tests, and routes.
- `security_review.md`
  - Canonical security document for the active FastAPI edge-service deployment.

## Dataset And Training Reference

- `annotation_workflow.md`
  - Reference workflow for dataset sampling and annotation.
  - Keep as background material; dataset work itself is already marked complete
    in `plan.md`.

## Notes

- The FastAPI edge service is the preferred MVP operator-facing surface.
- The ASP.NET `portal/` app still exists in the repo, but it is no longer the
  primary UI path for the current MVP.
- If two docs start covering the same operational task, prefer merging them
  here instead of keeping parallel runbooks.
