"""VisionGuard Safety Command Center (Streamlit).

Launch with:
    streamlit run src/visionguard/dashboard/app.py

Pick a video, run the analysis (or reopen a previous run), and explore:
annotated playback, live alert table, KPIs, incident timeline, danger heatmap,
and one-click PDF incident report export.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from visionguard.assistant.assistant import SafetyAssistant
from visionguard.pipeline import SafetyPipeline
from visionguard.reporting.pdf import IncidentReportBuilder
from visionguard.storage.event_store import EventStore, RunRecord
from visionguard.utils.config import AppConfig, load_config, resolve_output_path
from visionguard.utils.logger import setup_logging

st.set_page_config(
    page_title="VisionGuard Safety Command Center",
    page_icon="🦺",
    layout="wide",
)

_SEVERITY_BADGES = {"info": "🔵", "warning": "🟠", "critical": "🔴"}
_TYPE_LABELS = {
    "ppe_violation": "PPE violation",
    "zone_intrusion": "Zone intrusion",
    "zone_dwell": "Zone loitering",
    "fall": "Fall",
}


@st.cache_resource
def get_config() -> AppConfig:
    config = load_config()
    setup_logging(config.app.log_level, config.app.log_dir)
    return config


@st.cache_resource
def get_pipeline() -> SafetyPipeline:
    """The pipeline holds the loaded models — build it once per session."""
    return SafetyPipeline(get_config())


def list_videos(config: AppConfig) -> list[Path]:
    videos_dir = config.paths.data_dir / "videos"
    return sorted(videos_dir.glob("*.mp4")) if videos_dir.exists() else []


def run_label(run: RunRecord) -> str:
    status = "" if run.finished_at else " (incomplete)"
    return f"Run {run.id} — {Path(run.video_source).name}{status}"


# --------------------------------------------------------------------- #
# Sidebar: choose a run or start a new analysis
# --------------------------------------------------------------------- #
config = get_config()
store = EventStore(config.events.database_path)

st.sidebar.title("🦺 VisionGuard")
st.sidebar.caption("AI Safety Command Center")

videos = list_videos(config)
if videos:
    selected_video = st.sidebar.selectbox(
        "Video", videos, format_func=lambda p: p.name
    )
    if st.sidebar.button("▶ Run analysis", type="primary", width="stretch"):
        progress_bar = st.sidebar.progress(0.0, text="Analyzing…")

        def on_progress(done: int, total: int) -> None:
            progress_bar.progress(
                min(done / total, 1.0), text=f"Analyzing… {done}/{total} frames"
            )

        result = get_pipeline().run(source=str(selected_video), progress=on_progress)
        progress_bar.progress(1.0, text="Done")
        st.session_state["run_id"] = result.run_id
else:
    st.sidebar.info("Put an .mp4 in data/videos (or run scripts/download_assets.py).")

runs = store.list_runs()
if not runs:
    st.title("Safety Command Center")
    st.info("No analysis runs yet — pick a video in the sidebar and press *Run analysis*.")
    st.stop()

default_run = st.session_state.get("run_id", runs[0].id)
run = st.sidebar.selectbox(
    "Analysis run",
    runs,
    index=next((i for i, r in enumerate(runs) if r.id == default_run), 0),
    format_func=run_label,
)

events = store.events_for_run(run.id)
stats = run.stats

# --------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------- #
st.title("Safety Command Center")
st.caption(f"{run_label(run)} · analyzed {run.started_at[:19].replace('T', ' ')} UTC")

by_type = stats.get("events_by_type", {})
top_violation = (
    _TYPE_LABELS.get(max(by_type, key=by_type.get), "—") if by_type else "—"
)
unique = stats.get("unique_counts", {})

risk = stats.get("risk_score") or {}
peak_risk = risk.get("peak", 0)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Peak risk score", f"{peak_risk:.0f}/100")
k2.metric("PPE compliance", f"{(run.compliance_rate or 0) * 100:.1f}%")
k3.metric("Safety events", len(events))
k4.metric("Workers seen", unique.get("worker", 0))
k5.metric("Most dangerous zone", stats.get("most_dangerous_zone") or "—")
k6.metric("Top violation", top_violation)

# --------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------- #
video_tab, alerts_tab, timeline_tab, heatmap_tab, report_tab, assistant_tab = st.tabs(
    ["📹 Annotated video", "🚨 Alerts", "📈 Timeline", "🔥 Heatmap", "📄 Report",
     "🤖 Assistant"]
)

with video_tab:
    video_path = resolve_output_path(
        stats.get("annotated_video_h264") or stats.get("annotated_video")
    )
    if video_path and video_path.exists():
        st.video(str(video_path))
        st.caption(
            f"Processed at {stats.get('processing_fps', '?')} FPS · "
            f"{run.frames_processed or 0} frames"
        )
    else:
        st.warning("Annotated video not found on disk.")

with alerts_tab:
    if not events:
        st.success("No safety events detected in this run. 🎉")
    else:
        severities = st.multiselect(
            "Severity filter",
            ["critical", "warning", "info"],
            default=["critical", "warning", "info"],
        )
        filtered = [e for e in events if e.severity.value in severities]
        table = pd.DataFrame(
            {
                "Severity": [
                    f"{_SEVERITY_BADGES[e.severity.value]} {e.severity.value}"
                    for e in filtered
                ],
                "Time": [e.timestamp_str() for e in filtered],
                "Type": [_TYPE_LABELS.get(e.event_type.value, e.event_type.value)
                         for e in filtered],
                "Description": [e.description for e in filtered],
                "Confidence": [f"{e.confidence:.0%}" for e in filtered],
            }
        )
        st.dataframe(table, width="stretch", hide_index=True)

        with_shots = [e for e in filtered if e.screenshot_path]
        if with_shots:
            st.subheader("Evidence")
            chosen = st.selectbox(
                "Incident",
                with_shots,
                format_func=lambda e: f"[{e.timestamp_str()}] {e.description}",
            )
            evidence_path = resolve_output_path(chosen.screenshot_path)
            if evidence_path and evidence_path.exists():
                st.image(str(evidence_path), width="stretch")

with timeline_tab:
    risk_timeline = risk.get("timeline") or []
    if risk_timeline:
        risk_frame = pd.DataFrame(risk_timeline, columns=["Video time (s)", "Risk score"])
        risk_figure = px.area(
            risk_frame,
            x="Video time (s)",
            y="Risk score",
            title="Safety Risk Score (0–100) over the video",
            range_y=[0, 100],
        )
        st.plotly_chart(risk_figure, width="stretch")
    if not events:
        st.info("No events to plot.")
    else:
        frame = pd.DataFrame(
            {
                "Video time (s)": [e.video_time for e in events],
                "Type": [_TYPE_LABELS.get(e.event_type.value, e.event_type.value)
                         for e in events],
            }
        )
        figure = px.histogram(
            frame,
            x="Video time (s)",
            color="Type",
            nbins=40,
            title="Safety events over the video",
        )
        figure.update_layout(bargap=0.05)
        st.plotly_chart(figure, width="stretch")

with heatmap_tab:
    heatmap_path = resolve_output_path(stats.get("heatmap_image"))
    if heatmap_path and heatmap_path.exists():
        st.image(
            str(heatmap_path),
            caption="Worker position density (hot = heavily used areas)",
            width="stretch",
        )
    else:
        st.info("No heatmap for this run (no workers observed).")

with report_tab:
    st.write(
        "Generate a PDF incident report with the executive summary and "
        "screenshot evidence for every incident."
    )
    if st.button("Generate PDF report", type="primary"):
        builder = IncidentReportBuilder(store, config.output.reports_dir)
        pdf_path = builder.build(run.id)
        st.session_state["pdf_path"] = str(pdf_path)
        st.success(f"Report written to {pdf_path}")
    pdf_path = st.session_state.get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        st.download_button(
            "⬇ Download report",
            data=Path(pdf_path).read_bytes(),
            file_name=Path(pdf_path).name,
            mime="application/pdf",
        )

with assistant_tab:
    st.write(
        "Ask questions about this run's incidents in plain language — e.g. "
        "*How many helmet violations happened?*, *Which zone had the most "
        "incidents?*, *Summarize today's safety issues.*"
    )

    @st.cache_resource
    def get_assistant() -> SafetyAssistant:
        return SafetyAssistant(store, get_config().assistant)

    history_key = f"assistant_history_{run.id}"
    history = st.session_state.setdefault(history_key, [])

    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["text"])

    question = st.chat_input("Ask about this run's safety events…")
    if question:
        history.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Consulting the incident history…"):
                result = get_assistant().answer(question, run.id)
            st.markdown(result.text)
            if result.sources:
                with st.expander(f"Evidence ({len(result.sources)} incidents)"):
                    for source in result.sources:
                        st.markdown(
                            f"- `[{source.timestamp_str()}]` {source.description}"
                        )
        history.append({"role": "assistant", "text": result.text})
