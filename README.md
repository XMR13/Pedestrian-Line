Vehicle Subclass Line Counter
================================

## Deksripsi Project

Project ini memproses video di jalan dan akan menghitung berapa banyak **kendaraan tertentu** (mis. *truck, trailer, pickup*, dll — daftar kelas akan ditentukan kemudian) yang melewati garis virtual.

Selain menghitung banyak kendaraan yang melewati garis virtual tersebut, program ini juga menghitung berdasarkan arah kendaraan **(A->B) atau (B->A)**, serta menghitung berdasarkan **kelas kendaraan** (sesuai model yang digunakan).

Project ini direncanakan untuk memenuhi kriteria sebagai berikut:

- Fokus baru (overhaul):
  - Bukan untuk tracking person/pedestrian.
  - Hanya tracking + counting **kelas kendaraan target** (kelas kendaraan lain tidak dihitung).
  - Model yang tersedia saat ini belum cukup untuk klasifikasi spesifik tersebut, jadi langkah terdekat adalah **annotasi dataset** untuk melatih model baru.

-  Mudah digunakan dan modular..
- Bisa digunaan untuk aplikasi komersial (menggunakan aplikasi dan library dengan *license* permisif).

Pipeline inti dari project ini adalah sebagai berikut::

> video → detect objects → track objects → count line crossings →
> write annotated output video + print counts → Output into analytics dashbord
>

Ilustrasi cara program ini digunakan adalah sebagai berikut
![gambar cara kerja](docs/readme_png/cara_kerja.png)


Architecture (High Level)
-------------------------

Untuk arsitekturnya bisa dirujuk dari gambar berikut ini:
![gambar arsitektur](docs/readme_png/arch_new_modified.png)


Dependencies & Setup
--------------------

Requirements:

- Python 3.10+.
- `uv` as the Python package manager.

Install dependencies:

```bash
uv sync
```

Key runtime dependencies (from `pyproject.toml`):

- `opencv-python`
- `numpy`
- `onnxruntime` / `onnxruntime-gpu` (ONNX backend, default)
- (optional) `torch` Jika ingin menggunakan backend model `pytorch.` (already supported)


Model & Data Layout
-------------------

Repo ini tidak menyimpan model atau file biner yang besar, agra bisa menggunakan program ini, kamu harus mendownload model sendiri.

- ONNX model file:
  - Expected (placeholder): `Models/vehicle_subclasses.onnx`
  - You can change the path via `--model` or by editing `ModelConfig.model_path` in `pedestrian_line_counter/config.py`.
  - Penting: untuk backend berbasis model (`onnx`/`tensorrt`/`torch`) kamu **wajib** mengatur filter kelas target (`--class-ids`) supaya tidak menghitung kelas yang tidak relevan.
- Input videos:
  - Default: a sample file provided by yourself (see `IOConfig` in `config.py`).
  - You can override via `--input`.
- Output videos:
  - Default: `output.mp4` in the project root (see `IOConfig`).
  - You can override via `--output`.

Dataset & Annotasi (Priority)
-----------------------------

Karena target kelas kendaraan (truck/trailer/pickup/dll) adalah **custom**, langkah pertama adalah menyiapkan dataset dan melakukan annotasi bounding box.

- Panduan detail: lihat `docs/annotation_workflow.md`.
- Output yang direkomendasikan untuk training: format YOLO (images + labels + `data.yaml` berisi `names:`). Proses training bisa menggunakan model mana saja (tapi pada project ini harus sesuai dengan inference yang telah dibuat)
- Jika ingin otomatis mengambil kandidat gambar dari video (tanpa line counting), gunakan:

```bash
uv run python -m yolo_kitv2 label run \
  --mode candidates \
  --input media/input.mp4 \
  --output-dir data/candidates/input_mp4 \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --max-per-track 3 \
  --warmup-frames 5
```

Jika sudah punya model awal dan ingin **auto-label** video lalu koreksi di CVAT (COCO format),
script ini akan menyimpan frame **hanya saat kendaraan muncul** (mirip extract_candidates) dan
sekaligus menulis COCO labels.

```bash
uv run python -m yolo_kitv2 label run \
  --mode coco \
  --input media/input.mp4 \
  --output-dir data/auto_labels/input_mp4 \
  --model Models/vehicle_subclasses.onnx \
  --class-names Models/metadata.yaml \
  --min-seconds-between 1.0 \
  --max-per-track 3 \
  --warmup-frames 5
```

Tip: gunakan `--min-seconds-between` atau `--min-frames-between` untuk skip antar frame yang disimpan.

Untuk **auto-label folder gambar** (input = directory berisi images):

```bash
uv run python -m yolo_kitv2 label run \
  --mode coco \
  --input data/images_raw \
  --output-dir data/auto_labels/images_raw \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --class-names Models/metadata.yaml \
  --every-n 2
```

Untuk **gabung beberapa hasil auto-label** jadi satu folder COCO (siap CVAT):

```bash
uv run python -m yolo_kitv2 coco merge \
  --inputs data/auto_labels/set_a data/auto_labels/set_b \
  --output-dir data/auto_labels/merged
```

Jika CVAT **gagal menemukan images** saat import COCO (biasanya karena `images[].file_name` masih `images/<nama>.jpg`
tapi kamu upload gambar dalam folder yang flat), jalankan CVAT fix untuk membasename `file_name` dan memastikan
`category_id` mulai dari 1:

```bash
uv run python -m yolo_kitv2 coco cvat-fix \
  --dataset-dir data/auto_labels/merged \
  --in-place \
  --basename-file-names
```

Jika kamu **hapus gambar** secara manual, perbaiki COCO JSON:

```bash
uv run python -m yolo_kitv2 coco prune \
  --dataset-dir data/auto_labels/merged \
  --in-place
```


Kalau hanya punya `annotations.json` (tanpa struktur dataset lengkap) dan ingin **distribusi label saja**:

```bash
uv run python -m yolo_kitv2 dataset viz \
  --annotations data/auto_labels/merged/annotations.json \
  --distribution-only \
  --output-dir data/auto_labels/merged_viz_dist
```

Penggunaan sederhana
-----------

From WSL or a shell:

```bash
cd "/Pedestrian Line"
```

Jalankan dengan detektor onnx dan model path
yang ingin digunakan. Jika ingin model default bisa diatur melaui `config.py`

```bash
uv run python main.py \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --input media/input.mp4 \
  --output media/output_test.mp4 \
  --output-encoder ffmpeg \
  --output-crf 28 \
  --output-preset slow \
  --show
```

Apa yang dilakukan:

-  Load model ONNX ke `onnxruntime` (GPU jika tersedia, kalau tidak kembali ke penggunaan CPU). 
-  Menjalankan deteksi → Tracking → Menghitung untuk setiap frame 
-  Menggambar bounding box, garis virtual, dan secara live A→B / B→A.
-  Menyimpan video hasil anotasi ke `media/output_test.mp4`.
-  Memprint hasil ketika sudah selesai menjalankan program (video sudah terselesaikan)

Catatan ukuran output video:

- Jika `ffmpeg` tersedia di machine, gunakan `--output-encoder ffmpeg` agar ukuran video output jauh lebih kecil daripada writer OpenCV lama (`mp4v`).
- Default rekomendasi:
  - `--output-encoder ffmpeg`
  - `--output-crf 28`
  - `--output-preset slow`
- Jika `ffmpeg` tidak tersedia, gunakan `--output-encoder auto` agar program fallback ke OpenCV.

Logging per-crossing (filesystem-first)
--------------------------------------

Untuk kebutuhan dashboard/website (portal) nanti, kamu bisa menulis **event per crossing** ke disk (JSONL) dan (opsional) thumbnail per event:

```bash
python3 -m pedestrian_line_counter.main \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --input media/input.mp4 \
  --spool-dir data/traffic_runs \
  --site-id subang \
  --camera-id cam_01 \
  --report-csv \
  --report-name report.csv
```

Output (contoh):

- `data/traffic_runs/YYYY-MM-DD/<run_uid>/run.json`
- `data/traffic_runs/YYYY-MM-DD/<run_uid>/events.jsonl`
- `data/traffic_runs/YYYY-MM-DD/<run_uid>/report.csv`
- `data/traffic_runs/YYYY-MM-DD/<run_uid>/thumbs/<event_uid>.jpg`

`run.json` now also includes `health_summary` at the end of a run (reconnect cycles/attempts,
stall counts, reader dropped frames, reader read failures, and effective FPS) for daily report aggregation.

`report.csv` berisi 1 baris untuk setiap crossing event (default kolom utama: `event_no`, `timestamp_s`,
`vehicle_type`, `direction`, `notes`; plus kolom teknis opsional seperti `track_id`, `frame_index`,
`confidence`, dan `thumb_relpath`).

One-command mode (single process: detect + spool + upload)
-----------------------------------------------------------

Kalau kamu tidak mau menjalankan 2 proses terpisah, gunakan integrated uploader langsung dari `main.py`:

```bash
python3 -m pedestrian_line_counter.main \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --input media/input.mp4 \
  --spool-dir data/traffic_runs \
  --site-id subang \
  --camera-id cam_01 \
  --video-start 2026-02-24T08:00:00+07:00 \
  --portal-upload \
  --portal-api-base-url http://portal.local:5000 \
  --portal-api-key "$PORTAL_API_KEY"
```

Untuk live RTSP, cukup pakai flag yang sama (`--portal-upload`) dan `main.py` akan melakukan sync berkala selama proses berjalan:

```bash
python3 -m pedestrian_line_counter.main \
  --rtsp-url "rtsp://user:pass@camera-host:554/stream" \
  --spool-dir data/traffic_runs \
  --site-id subang \
  --camera-id cam_01 \
  --portal-upload \
  --portal-api-base-url http://portal.local:5000 \
  --portal-api-key "$PORTAL_API_KEY" \
  --portal-upload-interval-s 10
```

Untuk deployment production dengan satu proses (auto-restart + tuning performa + template systemd), pakai runbook:

- `docs/single_loop_production_runbook.md`
- launcher script: `scripts/run_single_loop_live.sh`

Jika kamu tidak ingin `export` setiap sesi, cukup isi `Portal.ApiKey` sekali di file lokal untracked `portal/appsettings.Local.json`.
Uploader (`portal_uploader.py`) dan integrated uploader (`main.py --portal-upload`) sekarang akan otomatis fallback ke file tersebut.

Portal website MVP (Phase 7.3)
------------------------------

Portal ASP.NET Core sekarang tersedia di folder `portal/`.

Mode default lokal: SQLite (biar cepat testing, tanpa setup SQL Server dulu).
Untuk deployment target tetap SQL Server.

Fitur MVP:

- Login gate minimal (cookie auth) dengan branding logo.
- Dashboard ringkasan total A→B/B→A + status review.
- Event browser dengan filter (site/camera/date/direction/class/review).
- Review queue cepat (Qualified Yes/No + notes + shortcut keyboard).
- Export CSV untuk event yang sudah direview.

Quick start:

```bash
cd portal
dotnet restore
dotnet run
```

Konfigurasi penting ada di `portal/appsettings.json`:

- `Database:Provider` (`Sqlite` default lokal, atau `SqlServer`).
- `ConnectionStrings:PortalDb` sesuai provider.
- `Portal:EvidenceRootPath` untuk penyimpanan thumbnail.
- `LoginGate:DisplayName` untuk nama tampilan reviewer.

`Portal:ApiKey` dan `LoginGate:Username/Password` jangan disimpan di file tracked.
Set via env var (`Portal__ApiKey`, `LoginGate__Username`, `LoginGate__Password`)
atau file lokal untracked `portal/appsettings.Local.json`.

Jika mode `Sqlite` default, DB `portal/portal.db` dibuat otomatis saat `dotnet run`.
Jika mode `SqlServer`, bisa pakai bootstrap:

- `portal/sql/001_init.sql`

Detail endpoint + setup ada di `portal/README.md`.

Runbook cepat (Windows)
-----------------------

Karena project ada di drive Windows, jalankan portal dari PowerShell Windows:

1. Start portal (SQLite lokal):

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
dotnet restore
dotnet run
```

Atau pakai script helper (background + log file):

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
.\scripts\start-portal.ps1 -Port 5000
```

2. Stop portal:

- Tekan `Ctrl + C` pada terminal yang sama.
- Jika masih ada proses di port 5000:

```powershell
$pid = (Get-NetTCPConnection -LocalPort 5000 -State Listen).OwningProcess
Stop-Process -Id $pid -Force
```

Atau pakai script helper:

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
.\scripts\stop-portal.ps1 -Port 5000 -Force
```

3. Simpan log ke file:

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
New-Item -ItemType Directory -Force logs | Out-Null
dotnet run *>&1 | Tee-Object -FilePath ".\logs\portal-$(Get-Date -Format yyyyMMdd-HHmmss).log"
```

Jika pakai `start-portal.ps1`, log otomatis dibuat di:

- `portal/logs/portal-<timestamp>-stdout.log`
- `portal/logs/portal-<timestamp>-stderr.log`

4. Jika port 5000 bentrok, pakai port lain:

```powershell
dotnet run --urls "http://localhost:5001"
```

Lalu sesuaikan uploader:

```bash
python3 -m pedestrian_line_counter.portal_uploader \
  --spool-dir data/traffic_runs \
  --api-base-url http://localhost:5001 \
  --api-key "$PORTAL_API_KEY"
```

Live RTSP Mode (Experimental)
-----------------------------

Next step dari project ini adalah menjalankan semua ini dengan RTSP live feed, terutama dengan menggunakan `--rtsp-url`. Direkomendasikan untuk menjalankannya seperti ini (tidak ada file output, periodic logging, serta pmebatasan fps)


Note: in live mode, writing is disabled by default unless you set `--output`
or a duration/frame limit, to avoid unbounded file growth.

Jetson Optimized RTSP Path (JetPack 5/6)
----------------------------------------

For Jetson devices, you can use OpenCV + GStreamer with NVDEC:

```bash
uv run python main.py \
  --rtsp-url "rtsp://user:pass@camera-host:554/stream" \
  --camera road_a \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --rtsp-capture-backend gstreamer \
  --rtsp-transport tcp \
  --rtsp-codec h264 \
  --rtsp-latency-ms 200 \
  --queue-policy drop_oldest \
  --target-fps 12 \
  --log-every-seconds 10 \
  --spool-dir data/traffic_runs \
  --no-write
```

Notes:

- `--queue-policy drop_oldest` keeps latency bounded for portal updates.
- `--queue-policy block` prioritizes completeness but can increase delay over time.
- `--rtsp-gst-pipeline` can override the generated pipeline (advanced tuning/debug).
- If GStreamer open fails, the app automatically falls back to OpenCV RTSP capture.

Jetson Optimized File Input Path (Preliminary)
----------------------------------------------

For offline video files on Jetson, you can now also ask OpenCV to try a
GStreamer file-input pipeline first:

```bash
uv run python main.py \
  --input media/testing_video2.mp4 \
  --camera road_a \
  --backend tensorrt \
  --model Models/model_fp16.engine \
  --class-names Models/metadata_vehicle.yaml \
  --input-capture-backend gstreamer \
  --write-processed-only \
  --fast-skip \
  --output media/output_contoh_v2.mp4
```

Notes:

- This preliminary file-mode path uses a generated `uridecodebin ... ! nvvidconv ... ! appsink` pipeline.
- `--input-gst-pipeline` can override the generated file-input pipeline for device-specific tuning.
- If the GStreamer file open fails, the app automatically falls back to the current OpenCV/FFmpeg file reader.

RTSP Reconnect Behavior
-----------------------

Reconnect policy is enabled by default in live mode and can be tuned from CLI:

```bash
uv run python main.py \
  --rtsp-url "rtsp://user:pass@camera-host:554/stream" \
  --camera road_a \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --rtsp-reconnect \
  --rtsp-reconnect-max-attempts 0 \
  --rtsp-reconnect-initial-delay 1.0 \
  --rtsp-reconnect-max-delay 30.0 \
  --rtsp-reconnect-backoff 2.0 \
  --rtsp-stall-timeout 5.0 \
  --no-write
```

Notes:

- `--rtsp-reconnect-max-attempts 0` means unlimited retries.
- Reconnect triggers when the reader stops or when no frame arrives for `--rtsp-stall-timeout`.
- On successful reconnect, the app clears transient tracker/counter per-track state to avoid stale IDs.
- Direction totals (`A->B`, `B->A`, and per-class direction totals) are preserved while the process keeps running.
