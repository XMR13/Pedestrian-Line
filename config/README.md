# Config Files

This folder contains versioned configuration data (JSON) and local-only overrides.

Recommended workflow:

- Keep `pedestrian_line_counter/config.py` as the *schema + sane defaults* (tracked in Git).
- Put production-ready, shareable configs under `config/` (tracked in Git).
- Put experimental configs under `config/local/` (ignored by Git).

**NOTES** : all the configurable config and what they for can be seen at `pedestrian_line_counter/config.py`. 

## Types of config
There's 6 types of config:
1. **Model Config**. This is the configuration of all the things that relate to the model (path, confidence score, iou, etc, etc). This configuration also correlates to the class itself (class_id).
2. **Track Config**. Configuration for trackig based on a simple greedy algorithm
3. **Line Config**. Configuration for line dividing the 2 sides. Things that can be changed : position, singular/multiplelines.
4. **I/O config**: Configuration for input and output processing
5. **Spool config**: Configuration for traffic event spool output.
6. **Visual config**: Configuration for drawing style and per-class box colors.

## Local Experiments (Ignored By Git)

Create a local override file like `config/local/exp.json` and run:

```bash
uv run python main.py --config config/local/exp.json ...
```

CLI flags still override values from `--config`.

Recommended for RTSP: keep credentials in `config/local/*.local.json` so they are never committed.
Prefer using environment variables inside configs, e.g.:

```json
{ "app": { "io": { "rtsp_url": "env:PLC_RTSP_URL" } } }
```

Then run with `PLC_RTSP_URL=...` in your shell environment.

You can also avoid passing the URL directly in CLI history:

```bash
PLC_RTSP_URL='rtsp://user:pass@host:554/stream' uv run python main.py --config config/local/exp.json --rtsp-url-env PLC_RTSP_URL
```

## Dumping A Production Config Snapshot

Once you have a setup you like, dump the resolved config to a tracked file:

```bash
uv run python main.py --config config/local/exp.json --dump-config config/cameras/camera_prod.json
```

Then use that JSON as the baseline for future runs (and keep iterating locally in `config/local/`).

## Example

See `config/overrides.example.json` for the supported shape.
See `config/subang.example.json` for a concrete camera-style example (includes line coords, RTSP placeholder, and spool metadata).

## Overridable Keys

Top-level structure for `--config`:

- `app.model`
- `app.tracker`
- `app.line`
- `app.io`
- `app.spool`
- `app.visual`

Main visual keys:

- `app.visual.track_color_by` (`"class"` or `"track"`)
- `app.visual.track_default_color` (BGR list, example: `[46, 204, 113]`)
- `app.visual.track_palette` (list of BGR colors)
- `app.visual.class_colors` (mapping: class_id string -> BGR color list)

Main RTSP reconnect keys (`app.io`):

- `app.io.rtsp_reconnect_enabled` (bool)
- `app.io.rtsp_reconnect_max_attempts` (int; `0` means unlimited)
- `app.io.rtsp_reconnect_initial_delay_s` (float > 0)
- `app.io.rtsp_reconnect_max_delay_s` (float >= initial delay)
- `app.io.rtsp_reconnect_backoff_factor` (float >= 1.0)
- `app.io.rtsp_stall_timeout_s` (float > 0)

Line source precedence:

1. CLI `--line-json` / `--camera`
2. Config `app.line.line_json_path` / `app.line.camera_name`
3. Config/default `app.line.start_norm` + `app.line.end_norm`
