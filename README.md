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

Untuk kebutuhan dashboard/operator UI, kamu bisa menulis **event per crossing** ke disk (JSONL) dan (opsional) thumbnail per event:

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

Single-process upload tetap tersedia di `main.py`, tetapi README ini sengaja
memfokuskan dokumentasi pada alur edge-service yang netral:

- pipeline menulis event ke `--spool-dir`
- uploader/service terpisah mengirim batch event ke backend IT

Pendekatan ini lebih cocok dengan arsitektur repo saat ini karena inference,
spool, API lokal, dan delivery worker sudah dipisahkan dengan jelas.

Untuk deployment production dengan satu proses (auto-restart + tuning performa + template systemd), pakai runbook:

- `docs/jetson_deployment_runbook.md`
- launcher script: `scripts/run_single_loop_live.sh`

Kalau backend IT belum tersedia, kamu bisa hardening uploader pakai mock backend lokal:

```bash
python3 scripts/mock_delivery_backend.py --port 18080 --state-dir tmp/mock_delivery_backend
```

Lalu arahkan uploader ke mock backend itu:

```bash
python3 -m pedestrian_line_counter.event_uploader \
  --spool-dir data/traffic_runs \
  --api-base-url http://127.0.0.1:18080 \
  --api-key local-dev
```

Request yang diterima mock backend akan disimpan ke `tmp/mock_delivery_backend/` supaya payload run/event/evidence bisa dicek tanpa menunggu backend team.

FastAPI edge service (Jetson-local hardened mode)
-------------------------------------------------

Untuk UI/operator view yang berjalan langsung dari repo Python ini, gunakan:

- `python3 -m pedestrian_line_counter.service`

Safe default sekarang adalah **loopback only**:

- service bind ke `127.0.0.1` / host loopback,
- FastAPI docs (`/docs`, `/redoc`, `/openapi.json`) tetap boleh aktif untuk local debugging,
- cocok untuk Jetson yang nanti diletakkan di balik reverse proxy / domain ketika keputusan infra sudah ada.

Contoh local-only:

```bash
python3 -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 127.0.0.1 \
  --port 8080 \
  --ui-password "replace-me" \
  --mutation-api-key "replace-me"
```

Kalau sementara harus diakses via **IP/LAN** karena domain belum ada, service juga bisa dijalankan dalam mode LAN yang lebih ketat:

```bash
python3 -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 0.0.0.0 \
  --port 8080 \
  --service-exposure lan \
  --no-service-docs \
  --service-trusted-host 127.0.0.1 \
  --service-trusted-host localhost \
  --service-trusted-host 192.168.1.50 \
  --ui-password "replace-me" \
  --mutation-api-key "replace-me"
```

Aturan hardening untuk mode LAN/IP:

- `--service-exposure lan` harus explicit.
- UI auth (`--ui-password`) wajib aktif.
- Mutation API key (`--mutation-api-key`) wajib aktif.
- Docs/OpenAPI wajib dimatikan (`--no-service-docs`).
- Trusted hosts wajib diisi explicit (`--service-trusted-host ...`).

Jadi untuk kondisi sekarang:

- belum ada domain: akses bisa lewat IP,
- auth perusahaan belum final: pakai UI login lokal sementara,
- nanti kalau domain / reverse proxy / auth final sudah ada, app ini tinggal dipasang di belakang layer tersebut tanpa ganti arsitektur inti.

Template deploy untuk systemd sekarang tersedia di:

- `deploy/systemd/pedestrian-edge-service.service.example`
- `deploy/systemd/pedestrian-edge-service.env.example`
- launcher script: `scripts/run_edge_service.sh`

Untuk panduan lengkap menjalankan **dua service** di Jetson:

- `docs/jetson_deployment_runbook.md`

Runbook ini menjelaskan:

- local video mode untuk validasi sekarang,
- RTSP mode untuk nanti,
- systemd setup untuk kedua service,
- env file mana yang harus diedit,
- cara verifikasi spool + UI end-to-end.

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

- `--queue-policy drop_oldest` keeps latency bounded for sync and UI updates.
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
