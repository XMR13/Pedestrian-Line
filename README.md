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

Untuk visualisasi label mirip dashboard (distribusi class, histogram box, dan sample overlay berlabel):

```bash
uv run python -m yolo_kitv2 dataset viz \
  --dataset-dir data/auto_labels/merged \
  --format coco \
  --output-dir data/auto_labels/merged_viz
```

Hasilnya ada di folder output: `summary.json`, `report.md`, chart PNG, dan folder `samples/`.

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
  --show
```

Apa yang dilakukan:

-  Load model ONNX ke `onnxruntime` (GPU jika tersedia, kalau tidak kembali ke penggunaan CPU). 
-  Menjalankan deteksi → Tracking → Menghitung untuk setiap frame 
-  Menggambar bounding box, garis virtual, dan secara live A→B / B→A.
-  Menyimpan video hasil anotasi ke `media/output_test.mp4`.
-  Memprint hasil ketika sudah selesai menjalankan program (video sudah terselesaikan)

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


Menentukan garis virtual untuk setiap kamera yang ada
---------------------------------------

Apabila terdapat lebih dari 1 buah kamera yang akan digunakan, pastinya garis - garis yang digunakan berpotensi terletak di tempat yang berbeda, maka dair itu, project ini juga bisa mengakomodir user untuk menentukan garisnya sendiri, dan menyimpannya kek dalam JSON

1. Pilih garis yang diigninakkn dan save:

```bash
uv run python -m pedestrian_line_counter.line_picker \
  --input media/input.mp4 \
  --lines 1 \
  --save config/line.json
```

Controls:

- Klik kiri: add points.
- `R` atau `C`: Reset titik.
- `Enter` / `Space`: Menyimpan titik dan garis yang telah ditentukan.
- `Esc` / `Q`: cancel.

2. Jalankan aplikasi tersebut dengan garis tadi:

```bash
uv run python main.py \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --input media/input.mp4 \
  --output media/output_with_line.mp4 \
  --line-json config/line.json \
  --show
```

Untuk menggunakan beberapa camera/multi camera
terutama yang disimpan di `config/camera/<line>`:

```bash
uv run python main.py \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --input media/road_a.mp4 \
  --output media/road_a_out.mp4 \
  --camera road_a
```


Backend dari pytorch ini mengininkan model dengan dimensi `(1, 3, H, W)` float tensor dengan range nilai berada di `[0,1]` dan akan mengembalikan output dengan shape `(N, 6)`. dengan `[x1, y1, x2, y2, score, class_id]`.

Jika model yang digunakan memiliki format yang berbeda, maka bisa diadaptasi dengan memodifikasi `pedestrian_line_counter/torch_detector.py`.



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
