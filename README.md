---
title: VisionGuard Safety AI
emoji: 🦺
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: "1.49.1"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Real-time CV workplace-safety command center
---

# VisionGuard Safety AI 🦺

**Real-Time Computer Vision Platform for Workplace Safety, PPE Compliance, and Hazard Intelligence** — live demo of the Safety Command Center.

Two pre-computed analysis runs are loaded so you can explore everything instantly:

- **Run 1 — construction site**: worker tracking, PPE compliance, restricted-zone alerts, worker–machinery proximity in real meters, danger heatmap
- **Run 2 — person down**: the full alarm chain — zone intrusion → missing PPE → **confirmed fall (critical)**

Try the tabs: annotated video, alerts with screenshot evidence, risk-score timeline, heatmap, one-click PDF report, and the AI assistant (ask *"what happened in this video?"*).

You can also press **Run analysis** to process a video fresh — note this Space runs on CPU, so it takes a few minutes (the project benchmarks 25–29 FPS on a laptop GPU).

Full source, docs, benchmarks, and tests: **[github.com/mohamed1232005/VisionGuard-Safety-AI](https://github.com/mohamed1232005/VisionGuard-Safety-AI)**
