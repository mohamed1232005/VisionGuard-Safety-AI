# VisionGuard Safety AI 🦺

**Real-Time Computer Vision Platform for Workplace Safety, PPE Compliance, and Hazard Intelligence**

VisionGuard turns ordinary CCTV and video streams into a real-time safety monitoring system for construction sites, factories, warehouses, and labs. Instead of simply detecting objects, it reasons about the scene — continuously answering one operational question: **"Is this workplace safe right now?"**

> **Status: Phase 1 (Core MVP) complete** — detection, tracking, PPE compliance, restricted zones, fall detection, command-center dashboard, and PDF incident reports, end to end.

---

## What it does (Phase 1)

| Feature | How it works |
|---------|--------------|
| **Multi-class detection** | YOLO11 fine-tuned on construction safety data detects workers, helmets, safety vests, their *absence* (`NO-Hardhat`, `NO-Safety Vest`), machinery, and vehicles |
| **Multi-object tracking** | ByteTrack assigns persistent IDs (`Worker #12`) and trajectories, surviving short occlusions via Kalman-filter motion prediction |
| **PPE compliance engine** | PPE evidence is associated to workers anatomically (helmets at head level, vests on the torso), smoothed over a rolling time window to kill detector flicker, and debounced with hysteresis + cooldown so one worker can't spam alerts |
| **Restricted zones** | Draw polygon zones on the camera view with an interactive editor; entry and dwell-time violations are detected using each worker's *ground point* (feet, not box overlap) |
| **Fall detection** | YOLO11-pose keypoints → torso-angle state machine; a fall is confirmed only when a person goes from upright to horizontal **and stays down** for N seconds |
| **Command center dashboard** | Streamlit app: KPI row, annotated video playback, filterable alert table with screenshot evidence, incident timeline, danger heatmap |
| **PDF incident reports** | One click generates an executive summary + per-incident evidence pages with screenshots and recommended actions |

Every safety event is persisted to a SQLite event store with screenshot evidence — the foundation for the Phase 2 RAG assistant.

## Roadmap

- [x] **Phase 1 — Core MVP**
- [x] **Phase 2 — Depth**: worker–vehicle proximity in real-world meters (camera calibration + ground-plane homography), 0–100 Safety Risk Score, RAG-based AI safety assistant over the event history
- [ ] **Phase 3 — Standout engineering**: cross-camera Re-Identification, ONNX/TensorRT edge optimization with measured FPS/latency/accuracy benchmarks, temporal behavior classifier

### Phase 2 highlights

| Feature | How it works |
|---------|--------------|
| **Calibrated proximity** | A ground-plane homography (calibrated once per camera with `scripts/calibrate_camera.py`) converts pixel positions to real-world meters — so "forklift within 1.5 m of Worker #1" means actual meters, not misleading pixel distance. Two debounced risk tiers (warning ≤5 m, critical ≤2 m) with per-pair cooldowns and near-miss counting |
| **Safety Risk Score (0–100)** | Every event adds its configured weight, decaying linearly over a rolling window — the score spikes on incidents and cools as the site behaves. Bands: 0–30 Safe · 31–60 Moderate · 61–80 High · 81–100 Critical. Shown live on the video HUD, plotted in the dashboard, summarized in the PDF |
| **AI Safety Assistant (RAG)** | Ask questions in plain language ("How many helmet violations today?"). Exact numbers come from SQL aggregates; relevant incidents come from semantic search (sentence-transformers + FAISS); Claude synthesizes the grounded answer. Works in retrieval-only mode without an API key |

**Proximity detection in action** (distance lines in real meters, critical pairs in red):

![Proximity alert](docs/images/proximity_alert.jpg)

## Architecture

```
Video / Camera
      │
      ▼
┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐
│ YOLO11        │──▶│ ByteTrack    │──▶│ Safety rule engines        │
│ detection     │   │ tracking     │   │  · PPE compliance          │
│ (+ YOLO-pose) │   │ (worker IDs) │   │  · Zone entry / dwell      │
└──────────────┘   └──────────────┘   │  · Fall state machine      │
                                      └─────────────┬──────────────┘
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │ Event store (SQLite) + screenshots  │
                              └───────┬─────────────────┬───────────┘
                                      ▼                 ▼
                         ┌────────────────────┐  ┌──────────────────┐
                         │ Streamlit command  │  │ PDF incident     │
                         │ center dashboard   │  │ reports          │
                         └────────────────────┘  └──────────────────┘
```

Design principles:

- **Model-agnostic core.** The detector translates raw model labels into a canonical taxonomy (`ObjectClass.WORKER`, `ObjectClass.NO_HELMET`, …); tracking, safety rules, storage, and reporting never see YOLO internals, so models can be swapped freely.
- **Config-driven.** Every threshold, path, and model choice lives in [`configs/config.yaml`](configs/config.yaml), parsed into frozen typed dataclasses that fail loudly on bad config.
- **Testable safety logic.** All rule engines operate on plain dataclasses — the 37-test suite runs in seconds with no GPU or model weights.

## Project structure

```
VisionGuard-Safety-AI/
├── configs/
│   ├── config.yaml            # All paths, thresholds, model settings
│   └── zones.json             # Restricted zones (drawn with the zone editor)
├── src/visionguard/
│   ├── detection/             # YOLO wrapper, pose estimation, core types
│   ├── tracking/              # ByteTrack wrapper (IDs, trajectories)
│   ├── safety/                # PPE / zones / falls / proximity / risk engines
│   ├── spatial/               # Ground-plane homography (pixels -> meters)
│   ├── assistant/             # RAG safety assistant (FAISS + Claude)
│   ├── storage/               # SQLite event store
│   ├── dashboard/             # Streamlit Safety Command Center
│   ├── reporting/             # PDF incident report builder
│   ├── utils/                 # Config, logging, video I/O, drawing
│   └── pipeline.py            # End-to-end orchestrator
├── scripts/
│   ├── download_assets.py     # Fetch model weights + sample video
│   ├── run_pipeline.py        # CLI analysis runner
│   ├── define_zones.py        # Interactive polygon zone editor
│   └── calibrate_camera.py    # Ground-plane calibration (pixels -> meters)
├── tests/                     # pytest suite (pure-logic, GPU-free)
├── data/                      # Videos & datasets (git-ignored)
├── models/                    # Model weights (git-ignored)
└── outputs/                   # Annotated videos, screenshots, heatmaps,
                               # reports, logs, events.db (git-ignored)
```

## Setup

Requires **Python 3.11+**. A CUDA GPU is strongly recommended (CPU works, slower).

```bash
git clone <repo-url>
cd VisionGuard-Safety-AI

python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)

pip install -r requirements.txt   # see GPU note below
pip install -e .

python scripts/download_assets.py # model weights + sample video
pytest                            # verify: all tests should pass
```

**GPU note:** `requirements.txt` pins the CUDA 12.1 PyTorch build, which needs the extra index URL:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Driver compatibility matters: NVIDIA driver 531+ supports the cu121 wheels. (On this project's dev machine — RTX 4050, driver 537.70 — the newer cu126 wheels failed with `CUDA error: device busy or unavailable`; cu121 resolved it.)

## Usage

**Analyze a video (CLI):**

```bash
python scripts/run_pipeline.py                          # config default video
python scripts/run_pipeline.py --source my_site.mp4    # specific file
python scripts/run_pipeline.py --source 0              # live webcam
```

Produces: annotated video + danger heatmap in `outputs/`, events + evidence screenshots in the event store, and a console summary.

**Draw restricted zones:**

```bash
python scripts/define_zones.py        # click vertices, N = finish zone, S = save
```

**Launch the Safety Command Center:**

```bash
streamlit run src/visionguard/dashboard/app.py
```

Run analyses from the sidebar, explore alerts/timeline/heatmap, export PDF reports.

## Results

Measured on an RTX 4050 Laptop GPU (6 GB), 1280×720 processing resolution, sample construction video (486 frames @ 50 fps source):

| Metric | Value |
|--------|-------|
| End-to-end processing speed | **25–29 FPS** (detection + pose + tracking + rules + annotation + video encode) |
| Detection / pose models | YOLO11s (PPE fine-tune) / YOLO11n-pose, both on CUDA |
| PPE compliance on sample video | 100% — correct: all workers wear helmet + vest (0 false alarms) |
| Test suite | 60 tests, ~3 s, no GPU required |

**Annotated output** — persistent worker IDs, PPE evidence, trajectory trails, live HUD:

![Annotated frame](docs/images/annotated_frame.jpg)

**Worker-position danger heatmap** (accumulated over the full video):

![Danger heatmap](docs/images/danger_heatmap.png)

## Models & attribution

| Model | Source | License |
|-------|--------|---------|
| PPE detection (YOLO11s) | [yihong1120/Construction-Hazard-Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection) | AGPL-3.0 |
| Pose (YOLO11n-pose) | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| Sample video | [Pexels — manas patra](https://www.pexels.com/video/workers-on-construction-11798561/) | Pexels License |

## Tech stack

Python 3.11 · PyTorch (CUDA) · Ultralytics YOLO11 · supervision (ByteTrack) · OpenCV · Streamlit · Plotly · ReportLab · SQLite · pytest
