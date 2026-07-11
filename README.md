# VisionGuard Safety AI 🦺 — Live Demo

Deployment bundle for the **VisionGuard Safety Command Center** (Streamlit Community Cloud).

Two pre-computed analysis runs are included so the dashboard is instantly explorable:

- **Run 1 — construction site**: worker tracking, PPE compliance, restricted-zone alerts, worker–machinery proximity in real meters, danger heatmap
- **Run 2 — person down**: the full alarm chain — zone intrusion → missing PPE → **confirmed fall (critical)**

Try the tabs: annotated video, alerts with screenshot evidence, risk-score timeline, heatmap, one-click PDF report, and the AI assistant.

Running a *fresh* analysis works too, but this demo runs on a small free CPU — expect a few minutes per video (the project benchmarks 25–29 FPS on a laptop GPU).

**Full source, documentation, benchmarks, and tests:**
👉 [github.com/mohamed1232005/VisionGuard-Safety-AI](https://github.com/mohamed1232005/VisionGuard-Safety-AI)
