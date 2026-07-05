# VisionGuard Safety AI

**Real-Time Computer Vision Platform for Workplace Safety, PPE Compliance, and Hazard Intelligence**

---

## Overview

VisionGuard Safety AI is an end-to-end computer vision platform that turns ordinary CCTV and video streams into a real-time safety monitoring system for construction sites, factories, warehouses, and labs. Instead of simply detecting objects, it *reasons about the scene* — answering one operational question continuously: **"Is this workplace safe right now?"**

The system detects PPE violations, restricted-zone intrusions, unsafe worker–vehicle proximity, falls, and hazardous behavior; tracks every worker and vehicle across time and across cameras; scores overall site risk from 0–100; and generates evidence-backed incident reports. A built-in AI assistant lets safety managers query the entire event history in natural language.

What separates VisionGuard from a typical detection demo is that it combines **real-world spatial reasoning** (camera calibration and ground-plane geometry, not naive pixel distance), **cross-camera re-identification**, **temporal behavior modeling**, and **edge-optimized deployment with measured performance** — the components that distinguish a production safety-analytics system from a student YOLO project.

> **One-line pitch:** *An AI safety command center that watches a worksite in real time, understands who is at risk and why, and produces the incident evidence to prove it.*

---

## What Makes This Project Outstanding

These are the components that elevate VisionGuard from "common PPE detector" to a system that signals real engineering depth. Every one of them is a concrete interview talking point.

1. **Metric-accurate proximity via camera calibration & homography.** Distances between workers and vehicles are computed in real-world meters by projecting the image onto the ground plane — not by measuring raw pixels. This is genuine computer vision, and it is the single feature that most portfolio versions get wrong.
2. **Cross-camera Re-Identification.** A worker keeps a consistent identity across multiple cameras and after occlusion, using appearance-embedding models. This is exactly what surveillance-analytics companies hire for.
3. **Temporal modeling for falls & unsafe behavior.** Instead of pure if-statements, a lightweight pose-sequence / video model classifies events over time, reducing false alarms and demonstrating sequence modeling — not just frame-by-frame detection.
4. **Edge-optimized deployment with real benchmarks.** Models are exported to ONNX/TensorRT and quantized, with measured FPS, latency, and accuracy trade-offs reported. This fills the deployment-engineering gap most portfolios lack.
5. **A quantified Safety Risk Score.** The system doesn't just list events — it synthesizes them into a single, interpretable site-risk metric, giving the project a unique identity.
6. **A RAG-powered Safety Assistant.** Natural-language querying over the incident database ties the CV system to modern LLM/RAG engineering.

---

## Core Feature Set

Structured for a tight, high-quality MVP first, then two depth phases. **Ship Phase 1 fully working before adding Phase 2 and 3** — a complete narrow system beats a broad broken one.

### Phase 1 — Core MVP (build this first, end-to-end)

**1. Multi-Class Detection**
- Worker detection
- PPE detection: helmet, safety vest (gloves / boots / mask optional)
- Vehicle detection: forklifts, trucks, cars, moving equipment

**2. Multi-Object Tracking**
- Persistent worker IDs and vehicle IDs across frames
- Trajectory tracking for each tracked entity
- Robust to short occlusions

**3. PPE Compliance Engine**
- "Worker without helmet / vest" alerts
- PPE compliance percentage per camera / per video
- Violation screenshots and violation timeline
- *Example alert:* `Worker #12 entered Zone A without helmet at 10:32 AM`

**4. Restricted-Zone Intrusion Detection**
- User draws polygon zones on the camera view (restricted / vehicle-only / machine-only / high-risk)
- Detects entry into and dwell time within a dangerous zone
- Per-zone violation counts and risk levels, with timestamped screenshot evidence

**5. Fall Detection**
- Pose estimation to detect sudden vertical-to-horizontal posture change
- Confirms a fall only if the person stays down for several seconds (reduces false positives)
- Emergency alert with before/after event clip and confidence score

**6. Safety Command Center Dashboard**
- Live camera feed with overlays
- Active alerts panel
- Key metrics: PPE compliance rate, unsafe events today, worker/vehicle counts, most dangerous zone, most common violation type
- Incident timeline and a heatmap of dangerous areas
- One-click report export

**7. PDF Incident Report Generator**
- Auto-generated reports containing: event type, timestamp, camera/location, involved worker/vehicle ID, confidence score, screenshot evidence, short event summary, and recommended action
- *Example title:* `Incident Report: PPE Violation and Restricted-Zone Entry`
- This is what makes the project feel like a real product.

### Phase 2 — Depth Upgrades (what makes it outstanding)

**8. Worker–Vehicle Proximity Risk (calibrated).**
- Camera calibration + homography to estimate **real-world distance** between workers and vehicles
- Risk levels (low / medium / high), near-miss detection, and saved event clips
- *Example alert:* `High-risk proximity: Forklift within unsafe distance of Worker #8`

**9. Safety Risk Score (0–100).**
- Aggregates PPE violations, zone intrusions, proximity risk, crowding, falls, and dangerous loitering into one interpretable score.

| Score | Meaning |
|-------|---------|
| 0–30 | Safe |
| 31–60 | Moderate Risk |
| 61–80 | High Risk |
| 81–100 | Critical Risk |

**10. AI Safety Assistant (RAG).**
- Natural-language querying over the event logs:
  - "How many helmet violations happened today?"
  - "Which zone had the most incidents?"
  - "Show me all high-risk forklift events."
  - "Generate a summary of today's safety issues."

### Phase 3 — Standout Engineering (optional, high-value)

**11. Cross-Camera Re-Identification.** Consistent worker identity across multiple cameras and after occlusion, using appearance embeddings.

**12. Edge Deployment & Optimization.** Export to ONNX/TensorRT, apply quantization, and publish a benchmark table: FPS, latency, and accuracy across configurations and hardware.

**13. Temporal Behavior Model.** Replace heuristic behavior rules with a small video/pose-sequence classifier for falls and 3–4 focused unsafe behaviors (running in restricted areas, standing under a suspended load, crossing a vehicle path, crowding in an unsafe zone).

---

## Tech Stack

| Part | Tools |
|------|-------|
| Detection | YOLOv8 / YOLO11 (optionally RT-DETR) |
| Tracking | ByteTrack / BoT-SORT |
| Pose / Fall | MediaPipe / YOLO-Pose |
| Segmentation (optional) | YOLO-Seg or SAM 2 |
| Re-ID (Phase 3) | Appearance-embedding Re-ID model |
| Spatial reasoning | OpenCV camera calibration + homography |
| Temporal model (Phase 3) | Pose-sequence / VideoMAE-style classifier |
| Backend | FastAPI |
| Dashboard | Streamlit (or React) |
| Database | PostgreSQL |
| Reports | ReportLab / WeasyPrint |
| AI Assistant | FAISS + SentenceTransformers + LLM (RAG) |
| Deployment | Docker + Docker Compose |
| Optimization | ONNX / TensorRT + quantization |
| Monitoring | Logs + FPS/latency dashboard |
| Advanced streaming (optional) | NVIDIA DeepStream |

---

## System Architecture

```
Camera / Video Upload
        ↓
Frame Processing
        ↓
YOLO Detection ── workers, helmets, vests, vehicles
        ↓
Tracking (ByteTrack) ── worker IDs, vehicle IDs, trajectories
        ↓
Spatial Reasoning ── homography → real-world distances & zones
        ↓
Safety Rules + Temporal Model ── PPE, zone intrusion, fall, proximity, behavior
        ↓
Event Database (PostgreSQL)
        ↓
Dashboard + PDF Reports ── live alerts, analytics, evidence
        ↓
AI Safety Assistant (RAG) ── natural-language queries over incidents
```

---

## Dataset Strategy

| Type | Purpose |
|------|---------|
| Public PPE dataset | Helmet / vest detection |
| Construction-safety dataset | Workers, equipment, machinery |
| COCO | General person / vehicle baseline |
| **Custom annotated video** | Makes the project unique and personal |
| Synthetic / staged scenarios | Rare events (falls, restricted-zone entry) |

Even a small custom-annotated set meaningfully raises the professionalism of the project and gives you a data-engineering story to tell.

---

## Suggested Repository Name

`VisionGuard-Safety-AI`

---

## CV Bullet Points This Project Produces

Use these (adapted) once it's built:

- *Built VisionGuard, a real-time computer vision safety platform detecting PPE violations, restricted-zone intrusions, falls, and worker–vehicle proximity risks across live video streams, with automated PDF incident reporting and a live analytics dashboard.*
- *Engineered metric-accurate proximity detection using camera calibration and ground-plane homography, and cross-camera worker Re-Identification for persistent tracking across occlusion.*
- *Deployed optimized detection models via ONNX/TensorRT with quantization, achieving [X] FPS at [Y] ms latency on [hardware]; containerized the full pipeline with Docker Compose.*
- *Integrated a RAG-based AI assistant over the incident database, enabling natural-language safety queries and automated daily summaries.*
