# EZVIZ Windows Bridge to AI Box RTSP Runbook

This document describes the end-to-end setup for rebroadcasting an EZVIZ camera
view from a Windows PC and exposing it as an RTSP stream that the AI box can
consume.

This runbook is written for beginners.
If you just want the fastest path, read **Section 5A - Quick Start (Tested Working Path)** first.

This design is intended for the current constraint:

- the EZVIZ camera is not directly accessible by the AI box,
- a Windows PC inside the office network can already view the camera through
  the EZVIZ app after receiving admin permission,
- the AI box and the Windows PC are on the same network,
- the AI pipeline is already designed to consume RTSP input.

Current project decision:

- this bridge is the approved MVP input path for immediate validation,
- the goal right now is to prove the end-to-end system works as expected,
- hardening work such as auto-restart, stricter firewall rules, and dedicated
  bridge-machine operating procedures will come after the MVP is proven.

Operational note:

- the team is intentionally using the paid EZVIZ subscription tier during this
  MVP phase so free-tier access limits do not block testing.

## 1. Final architecture

```text
Remote camera
  -> office CCTV access / EZVIZ account permission
  -> EZVIZ app on Windows bridge PC
  -> OBS Window Capture
  -> OBS publishes local RTMP
  -> MediaMTX on the same Windows PC
  -> MediaMTX exposes RTSP over LAN
  -> AI box reads RTSP and runs detection / tracking / counting
```

Important design choice:

- `MediaMTX` should run on the Windows bridge PC, not on the AI box, for this setup.

Why:

- the Windows PC is the source of the captured video,
- OBS can publish into a local MediaMTX instance on the same machine,
- the AI box then reads one stable RTSP URL over the LAN,
- this keeps the AI box simple and aligned with the current RTSP-based pipeline.

## 2. Required hardware

You only need two active devices for this bridge design.

### 2.1 AI box

Purpose:

- runs the existing pedestrian line counter pipeline,
- consumes the RTSP stream coming from the Windows bridge PC.

Minimum practical requirements:

- Jetson Orin NX / reComputer J4012 target is acceptable,
- wired Ethernet recommended,
- enough storage for spool output and logs,
- the AI box must be able to reach the Windows PC IP over the local network.

### 2.2 Intermediate hardware: Windows bridge PC

Purpose:

- logs into the EZVIZ account,
- displays the live camera in the EZVIZ application,
- captures the live-view window,
- republishes that view as RTSP for the AI box.

Recommended practical requirements:

- Windows 10 or Windows 11,
- wired Ethernet preferred over Wi-Fi,
- modern CPU with hardware H.264 encoder support if available,
- 8 GB RAM minimum,
- screen resolution stable during operation,
- power settings configured so the machine does not sleep.

This PC should be treated as a dedicated bridge machine when in use.

## 3. Required software on the Windows PC

Install the following:

1. `EZVIZ Studio` or the approved EZVIZ desktop application that can display the target camera.
2. `OBS Studio`
3. `MediaMTX`

Optional but recommended:

- a remote-management tool or Windows auto-login policy if this machine must recover after reboot,
- GPU driver updates if OBS hardware encoding is used.

Repo-managed helper assets for this bridge now live in:

- `deploy/windows_bridge/README.md`
- `deploy/windows_bridge/mediamtx.yml.example`
- `deploy/windows_bridge/start_mediamtx.ps1`
- `deploy/windows_bridge/open_rtsp_firewall.ps1`

## 4. Output expected by the AI box

The final output of this bridge is a normal RTSP stream on the LAN.

Recommended final stream URL format:

```text
rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01
```

Example:

```text
rtsp://192.168.1.25:8554/live/ezviz_cam01
```

This is the only URL the AI box should need.

## 5. End-to-end setup steps

## 5A. Quick Start (Tested Working Path)

This is the shortest version of the setup that was already proven to work.

### Windows bridge PC

1. Install and open:
   - EZVIZ
   - OBS Studio
   - MediaMTX
2. Put `mediamtx.yml` in the MediaMTX folder.
3. Start `mediamtx.exe`.
4. In OBS:
   - add the EZVIZ window as `Window Capture`
   - crop it so only the camera image is visible
   - set `Stream -> Service` to `Custom`
   - set `Server` to:

```text
rtmp://127.0.0.1/live/ezviz_cam01
```

   - leave `Stream Key` empty
   - set FPS to `10`
   - click `Start Streaming`
5. Open Windows PowerShell as Administrator and allow the RTSP port:

```powershell
New-NetFirewallRule `
  -DisplayName "Pedestrian Line RTSP 8554" `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort 8554
```

6. Check the Windows machine LAN IP with:

```powershell
ipconfig
```

### What success looks like in MediaMTX

When OBS is streaming correctly, MediaMTX should show lines similar to:

```text
[RTSP] listener opened on :8554
[RTMP] listener opened on :1935
[path live/ezviz_cam01] stream is available and online
[RTMP] ... is publishing to path 'live/ezviz_cam01'
```

If you see those lines, Windows-side setup is usually correct.

### AI box

From the project root on the AI box, run:

```bash
PYTHONPATH=. python3 scripts/check_rtsp.py \
  --url rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01 \
  --backend opencv
```

Expected success output:

- `opened: True`
- `decoded_frames:` greater than `0`
- `result: OK`

Example of a real successful probe:

```text
url: rtsp://192.168.1.10:8554/live/ezviz_cam01
opened: True
backend: FFMPEG
reported_resolution: 1920x1080
reported_fps: 10.0
decoded_frames: 10
first_frame_shape: 1920x1080
result: OK
```

If you get `opened: False`, the most common cause is Windows Firewall not allowing `8554`.

### 5.1 Prepare the Windows bridge PC

1. Put the Windows PC on the same LAN as the AI box.
2. Assign it a stable IP address if possible.
3. Disable sleep, hibernate, and automatic screen-off while the bridge is operating.
4. Disable Windows notifications or Focus Assist popups that could cover the EZVIZ window.
5. Make sure the PC can stay logged in while the bridge is running.

Recommended Windows settings:

- Sleep: `Never`
- Screen off: `Never` while operating
- Focus Assist: `Priority only` or `Alarms only`
- Windows Update active hours set so the machine does not reboot unexpectedly

### 5.2 Set up EZVIZ access

1. Create the EZVIZ account that will be used by the bridge PC.
2. Ask the CCTV admin to grant that account permission to the target camera.
3. Log into EZVIZ on the Windows PC.
4. Verify that the live feed opens reliably.
5. Put the target camera into a consistent single-camera view.

Operator rule:

- do not move the window around after OBS cropping is configured,
- do not switch layouts or open side panels unless OBS is updated too.

### 5.3 Install and start MediaMTX on the Windows PC

Use MediaMTX as the local media relay on the Windows bridge PC.

Recommended folder example:

```text
C:\mediamtx\
```

Any folder is acceptable as long as:

- `mediamtx.exe` and `mediamtx.yml` are in the same folder,
- you start `mediamtx.exe` from that folder.

Real example that was used successfully:

```text
R:\APLIKASI\mediamtx_v1.17.1_windows_amd64\
```

Place the MediaMTX files there and copy the repo example config:

```text
deploy/windows_bridge/mediamtx.yml.example
-> C:\mediamtx\mediamtx.yml
```

The example config keeps the MVP setup intentionally small:

- RTMP enabled on `tcp/1935` for local OBS publishing
- RTSP enabled on `tcp/8554` for LAN readers
- RTSP transport forced to TCP for simpler firewall behavior
- HLS, WebRTC, SRT, API, and metrics left disabled for the first bridge pass
- explicit `live/ezviz_cam01` publisher path

Recommended helper-script location:

```text
C:\pedline-bridge\
```

Copy these repo files there:

```text
deploy/windows_bridge/start_mediamtx.ps1
deploy/windows_bridge/open_rtsp_firewall.ps1
```

Start MediaMTX from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File C:\pedline-bridge\start_mediamtx.ps1
```

If you use a different MediaMTX folder, pass it explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File C:\pedline-bridge\start_mediamtx.ps1 -MediaMTXRoot "R:\APLIKASI\mediamtx_v1.17.1_windows_amd64"
```

You can also start it manually:

```powershell
cd "R:\APLIKASI\mediamtx_v1.17.1_windows_amd64"
.\mediamtx.exe
```

By default, the RTSP service listens on:

```text
tcp/8554
```

The local OBS publisher writes into:

```text
tcp/1935
```

That RTMP listener only needs to be reachable from the same Windows PC.

### 5.4 Allow the RTSP port through Windows Firewall

Allow inbound access to the MediaMTX RTSP port from the local network.

Minimum required inbound rule:

- TCP `8554`

If you want to keep it narrow, allow only the AI box IP to reach that port.

Recommended way to add the rule:

```powershell
powershell -ExecutionPolicy Bypass -File C:\pedline-bridge\open_rtsp_firewall.ps1
```

Manual firewall command that was proven to work:

```powershell
New-NetFirewallRule `
  -DisplayName "Pedestrian Line RTSP 8554" `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort 8554
```

If the AI box IP is already fixed, narrow it immediately:

```powershell
powershell -ExecutionPolicy Bypass -File C:\pedline-bridge\open_rtsp_firewall.ps1 -RemoteAddress 192.168.1.50
```

### 5.5 Configure OBS to capture the EZVIZ live view

1. Open `OBS Studio`.
2. Create a new scene, for example:

```text
EZVIZ Bridge
```

3. Add a `Window Capture` source and select the EZVIZ window.
4. Crop the source so only the live video area remains.
5. Remove unnecessary UI regions:
   - sidebars,
   - playback controls,
   - account info,
   - menus,
   - extra borders.
6. Keep the camera feed at a stable size.

Strong recommendation:

- fullscreen the camera view inside EZVIZ if possible before cropping in OBS.
- do not move or resize the EZVIZ window after OBS cropping is correct.

### 5.6 Configure OBS to publish into local MediaMTX

In OBS, use a custom RTMP stream target pointing to the MediaMTX instance on
the same Windows PC.

Recommended OBS streaming target:

- Service: `Custom`
- Server:

```text
rtmp://127.0.0.1/live/ezviz_cam01
```

- Stream Key:

```text
(leave empty)
```

This matches the MediaMTX publisher path in the repo example config and
produces a final MediaMTX path of:

```text
/live/ezviz_cam01
```

Recommended OBS output settings for the first working version:

- Encoder: `H.264`
- Frame rate: `10` to `15 FPS`
- Resolution: start with the captured window size, then reduce only if needed
- Audio: disable if not needed
- Bitrate: start around `1500` to `3000 Kbps`
- Keyframe interval: `2`

Priority for this bridge:

- stability first,
- consistent framing second,
- low latency third,
- perfect visual quality is not the goal.

Practical recommendation:

- start with `10 FPS`
- leave audio enabled only if you truly need it
- if you do not need audio, disable it in OBS to keep the stream simpler

### 5.7 Start the bridge stream

1. Start MediaMTX on the Windows PC.
2. Open EZVIZ and display the target camera live view.
3. Open OBS.
4. Confirm the cropped camera video looks correct inside OBS.
5. Start streaming from OBS.

At this point, MediaMTX should expose the RTSP stream on the LAN.

### 5.8 Verify the RTSP output from another machine

Before pointing the AI box at it, verify that the RTSP stream works.

Use VLC, FFmpeg, or OpenCV from another machine on the same LAN.

Recommended quick checks:

- On the Windows bridge PC:
  - verify MediaMTX is still running,
  - confirm OBS shows `LIVE`,
  - optionally check `netstat -ano | findstr :8554`
- From the AI box or another LAN machine:
  - open the RTSP URL in VLC, or
  - use `ffplay rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01`

Recommended AI box probe from this repo:

```bash
PYTHONPATH=. python3 scripts/check_rtsp.py \
  --url rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01 \
  --backend opencv
```

Important:

- run that command from the project root on the AI box,
- keep `PYTHONPATH=.` in front so Python can import the local package cleanly.

If the AI box is the Jetson target and you want to test the same capture path
planned for production, probe with GStreamer first:

```bash
PYTHONPATH=. python3 scripts/check_rtsp.py \
  --url rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01 \
  --backend gstreamer \
  --transport tcp \
  --codec h264 \
  --latency-ms 120
```

Success criteria for the probe:

- `opened: True`
- `decoded_frames:` greater than `0`
- `result: OK`

If `opencv` works but `gstreamer` fails on the AI box:

- keep the bridge running,
- continue with `opencv` first for connectivity proof,
- then inspect Jetson GStreamer plugin/runtime issues separately.

If both `opencv` and `gstreamer` fail:

- confirm the Windows IP is correct,
- confirm OBS is still streaming,
- confirm MediaMTX still shows the stream path as online,
- confirm Windows Firewall rule for `8554` exists.

Expected RTSP URL:

```text
rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01
```

If the test machine can open the stream, the bridge is working.

### 5.9 Configure the AI box to consume the stream

Use the bridge RTSP URL in the AI box runtime configuration.

Example:

```env
PLC_INPUT_PATH=
PLC_RTSP_URL=rtsp://192.168.1.25:8554/live/ezviz_cam01
```

Recommended related runtime options for the first test:

```env
PLC_RTSP_TRANSPORT=tcp
PLC_TARGET_FPS=12
PLC_FRAME_STRIDE=1
PLC_QUEUE_SIZE=3
PLC_QUEUE_POLICY=drop_oldest
```

Recommended first AI box smoke test before switching the systemd service:

```bash
export PLC_RTSP_URL=rtsp://192.168.1.25:8554/live/ezviz_cam01
python3 -m pedestrian_line_counter.main \
  --rtsp-url-env PLC_RTSP_URL \
  --camera-id cam_01 \
  --site-id subang \
  --max-seconds 30 \
  --target-fps 12 \
  --frame-stride 1 \
  --headless-status-json /tmp/pedline_rtsp_status.json \
  --log-every-seconds 5 \
  --no-write
```

For the Jetson runtime path that matches the service wrapper more closely, use:

```bash
export PLC_RTSP_URL=rtsp://192.168.1.25:8554/live/ezviz_cam01
export PLC_RTSP_CAPTURE_BACKEND=gstreamer
export PLC_RTSP_TRANSPORT=tcp
export PLC_RTSP_CODEC=h264
export PLC_RTSP_LATENCY_MS=120
export PLC_TARGET_FPS=12
export PLC_FRAME_STRIDE=1
bash scripts/run_single_loop_live.sh
```

For the first smoke test, temporarily add:

```env
PLC_MAX_SECONDS=30
```

Then remove that limit once the stream is confirmed healthy.

Beginner tip:

- do not switch the real systemd service immediately.
- first make sure the probe works,
- then make sure the short smoke test works,
- only then update `single_loop.service`.

If the Jetson deployment wrapper is already in use, this bridge URL simply
replaces the earlier placeholder RTSP URL.

## 6. Final stream contract

The AI box should consume exactly one stable endpoint.

### Recommended production-like endpoint

```text
rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01
```

### Ownership of each stage

- EZVIZ app:
  provides the operator-approved camera view
- OBS:
  captures the view and encodes it
- MediaMTX:
  repackages the captured feed as a standard RTSP stream
- AI box:
  reads RTSP and runs the existing counting pipeline

## 7. Recommended operating procedure

Each time the bridge is started:

1. Start the Windows PC.
2. Start MediaMTX.
3. Open EZVIZ and confirm the target camera is visible.
4. Open OBS and check the crop.
5. Start OBS streaming.
6. Verify the RTSP URL from the AI box.
7. Start the AI service.
8. Confirm a new spool run is created on the AI box.
9. Open the FastAPI dashboard and confirm new runtime data appears.

Recommended long-term improvement:

- set the Windows PC to auto-start MediaMTX and OBS after login,
- keep a dedicated OBS profile for this bridge,
- keep the EZVIZ layout fixed and document it.

Recommended MVP operator shortcut:

- keep `C:\mediamtx\` for MediaMTX binaries and config,
- keep `C:\pedline-bridge\` for helper scripts only,
- use a dedicated OBS scene named `EZVIZ Bridge`,
- keep the OBS server field fixed at:

```text
rtmp://127.0.0.1/live/ezviz_cam01
```

## 8. MVP validation checklist

Use this checklist for the first manager-facing proof that the deployment works.

The MVP run is considered successful when all of the following are true in the
same session:

1. The Windows bridge PC opens the target camera reliably in EZVIZ.
2. OBS captures only the intended live-view area.
3. MediaMTX exposes a reachable LAN RTSP stream:

```text
rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01
```

4. The AI box opens that RTSP stream using config only.
5. `single_loop.service` keeps processing frames continuously.
6. The spool directory receives a new run and expected event artifacts.
7. `edge_service.service` continues to show health, recent runs, and recent
   events correctly.
8. The same flow can be repeated for at least `2` to `3` runs without
   unexpected manual fixes.

Recommended proof package for the MVP review:

- screenshot of the EZVIZ window on the bridge PC,
- screenshot of the OBS preview/crop,
- proof that the RTSP URL opens from another machine or the AI box,
- screenshot of the FastAPI dashboard while the run is active,
- one sample spool run directory showing `run.json`, `events.jsonl`, and
  evidence files,
- short run notes describing total runtime and whether any restart or reconnect
  was needed.

## 9. Failure points and risks

This design works, but it is still a screen-capture bridge, not a native
camera stream.

Main risks:

- EZVIZ app logs out or loses permission
- EZVIZ changes the window layout
- a popup or notification covers the video
- the window is moved or resized
- Windows sleeps or reboots
- OBS stops streaming
- MediaMTX is not running
- Windows firewall blocks port `8554`

Operational consequences:

- the AI box may still connect to RTSP but receive a broken or black frame,
- detections may degrade if menus or overlays enter the captured region.

## 10. Troubleshooting checklist

### AI box cannot open RTSP

Check:

- MediaMTX is running on the Windows PC
- Windows Firewall rule for `8554` actually exists
- Windows Firewall allows inbound `8554`
- the Windows PC IP address did not change
- OBS is actually streaming
- the stream path is correct:

```text
/live/ezviz_cam01
```

Fast Windows-side checks:

```powershell
Get-NetFirewallRule -DisplayName "Pedestrian Line RTSP 8554"
ipconfig
```

Fast AI-box check:

```bash
PYTHONPATH=. python3 scripts/check_rtsp.py --url rtsp://<WINDOWS_PC_IP>:8554/live/ezviz_cam01 --backend opencv
```

### RTSP opens but the picture is wrong

Check:

- EZVIZ is showing the correct camera
- the EZVIZ window size/layout did not change
- OBS crop still matches the live-view area
- no menu, popup, or notification is covering the image

### Stream is too slow or unstable

Try:

- lower OBS FPS from `15` to `10`
- reduce output bitrate
- use wired Ethernet on both devices
- close other CPU-heavy apps on the Windows PC
- use hardware H.264 encoding if available

## 11. Minimum deliverable for this project stage

This setup should be considered successful when all of the following are true:

1. The EZVIZ account can open the target camera on the Windows PC.
2. OBS can capture only the useful live-view region.
3. MediaMTX exposes a stable RTSP stream on the LAN.
4. The AI box can open that RTSP stream continuously.
5. The existing AI pipeline can process the stream without code changes other
   than the RTSP URL and runtime tuning.

Current validation status:

- Windows bridge publishing to MediaMTX: verified
- AI box OpenCV/FFMPEG probe opening the bridged RTSP URL: verified
- Example verified result:
  - `opened: True`
  - `decoded_frames: 10`
  - `reported_resolution: 1920x1080`
  - `reported_fps: 10.0`
  - `result: OK`

## 12. Recommended next step after this document

After the bridge works end to end, validate in this order:

1. RTSP opens successfully on the AI box.
2. The AI box processes frames continuously for at least 30 to 60 minutes.
3. Spool output is written correctly.
4. FastAPI status/UI still behaves correctly while using the bridged RTSP source.
5. Only after that, begin accuracy tuning on the captured EZVIZ view.

After the MVP is accepted, the next hardening backlog should include:

- auto-start and recovery for MediaMTX and OBS,
- firewall narrowing so only the AI box can read the RTSP port,
- a dedicated Windows bridge account/profile,
- a documented startup and recovery SOP for the bridge PC,
- longer soak tests before calling the bridge operationally stable.
