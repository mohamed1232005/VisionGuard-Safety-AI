"""PDF incident report generation (ReportLab).

Turns a finished analysis run into a professional report: an executive summary
with the key safety metrics, followed by one evidence section per incident
(screenshot, details table, recommended action).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from visionguard.safety.events import SafetyEvent, Severity
from visionguard.storage.event_store import EventStore, RunRecord

logger = logging.getLogger(__name__)

_BRAND = colors.HexColor("#1a3c5e")
_SEVERITY_COLORS = {
    Severity.INFO: colors.HexColor("#6c757d"),
    Severity.WARNING: colors.HexColor("#cc7a00"),
    Severity.CRITICAL: colors.HexColor("#b30000"),
}


class IncidentReportBuilder:
    """Builds one PDF report per analysis run."""

    def __init__(self, store: EventStore, reports_dir: Path) -> None:
        self._store = store
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        base = getSampleStyleSheet()
        self._styles = {
            "title": ParagraphStyle(
                "vg-title", parent=base["Title"], textColor=_BRAND, fontSize=22
            ),
            "h2": ParagraphStyle(
                "vg-h2", parent=base["Heading2"], textColor=_BRAND
            ),
            "body": base["BodyText"],
            "small": ParagraphStyle(
                "vg-small", parent=base["BodyText"], fontSize=8,
                textColor=colors.grey,
            ),
        }

    def build(self, run_id: int) -> Path:
        """Generate the report for a run and return the PDF path."""
        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} does not exist")
        events = self._store.events_for_run(run_id)

        path = self._reports_dir / f"incident_report_run_{run_id}.pdf"
        document = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            title=f"VisionGuard Incident Report — Run {run_id}",
            author="VisionGuard Safety AI",
        )

        story: list = [*self._summary_section(run, events)]
        if events:
            story.append(PageBreak())
            story.append(Paragraph("Incident Details", self._styles["title"]))
            story.append(Spacer(1, 0.4 * cm))
            for index, event in enumerate(events, start=1):
                story.extend(self._incident_section(index, event))

        document.build(story)
        logger.info("Report for run %d written to %s", run_id, path)
        return path

    # ------------------------------------------------------------------ #
    # Sections
    # ------------------------------------------------------------------ #
    def _summary_section(self, run: RunRecord, events: list[SafetyEvent]) -> list:
        stats = run.stats
        critical = sum(1 for e in events if e.severity is Severity.CRITICAL)
        by_type = stats.get("events_by_type", {})
        top_violation = max(by_type, key=by_type.get) if by_type else "none"
        unique = stats.get("unique_counts", {})

        rows = [
            ["Video source", str(run.video_source)],
            ["Analyzed at (UTC)", run.started_at[:19].replace("T", " ")],
            ["Video analyzed", f"{run.duration_seconds or 0:.0f} s "
                               f"({run.frames_processed or 0} frames)"],
            ["PPE compliance rate", f"{(run.compliance_rate or 0) * 100:.1f}%"],
            ["Total safety events", f"{len(events)} ({critical} critical)"],
            ["Most common violation", top_violation.replace("_", " ")],
            ["Most dangerous zone", str(stats.get("most_dangerous_zone") or "n/a")],
            ["Workers observed", str(unique.get("worker", 0))],
            ["Vehicles / machinery", f"{unique.get('vehicle', 0)} / "
                                     f"{unique.get('machinery', 0)}"],
            ["Falls detected", str(stats.get("falls_detected", 0))],
        ]
        risk = stats.get("risk_score") or {}
        if risk:
            rows.append(
                ["Peak risk score",
                 f"{risk.get('peak', 0):.0f}/100 at {risk.get('peak_time', 0)}s"]
            )
        proximity = stats.get("proximity")
        if proximity:
            min_distance = proximity.get("min_distance_m")
            rows.append(
                ["Worker-vehicle near misses",
                 f"{proximity.get('near_misses', 0)} "
                 f"(closest: {min_distance} m)" if min_distance is not None
                 else str(proximity.get('near_misses', 0))]
            )
        table = Table(rows, colWidths=[6 * cm, 10 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                    ("TEXTCOLOR", (0, 0), (0, -1), _BRAND),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d4e0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return [
            Paragraph("VisionGuard Safety AI", self._styles["title"]),
            Paragraph(f"Incident Report — Run {run.id}", self._styles["h2"]),
            Paragraph(
                f"Generated {datetime.now():%Y-%m-%d %H:%M}", self._styles["small"]
            ),
            Spacer(1, 0.6 * cm),
            Paragraph("Executive Summary", self._styles["h2"]),
            table,
        ]

    def _incident_section(self, index: int, event: SafetyEvent) -> list:
        severity_color = _SEVERITY_COLORS[event.severity]
        heading = Paragraph(
            f"Incident {index}: "
            f"{event.event_type.value.replace('_', ' ').title()} "
            f"<font color='#{severity_color.hexval()[2:]}'>"
            f"[{event.severity.value.upper()}]</font>",
            self._styles["h2"],
        )

        rows = [
            ["Video timestamp", event.timestamp_str()],
            ["Real time (UTC)", event.wall_time.strftime("%Y-%m-%d %H:%M:%S")],
            ["Involved", event.track_label or "unknown"],
            ["Zone", event.zone_name or "—"],
            ["Confidence", f"{event.confidence:.0%}"],
            ["Summary", event.description],
        ]
        details = Table(rows, colWidths=[4 * cm, 12 * cm])
        details.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d4e0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )

        section: list = [heading, details, Spacer(1, 0.3 * cm)]
        screenshot = Path(event.screenshot_path) if event.screenshot_path else None
        if screenshot and screenshot.exists():
            section.append(Image(str(screenshot), width=15 * cm, height=8.4 * cm))
            section.append(Spacer(1, 0.3 * cm))
        section.append(
            Paragraph(
                f"<b>Recommended action:</b> {event.recommended_action}",
                self._styles["body"],
            )
        )
        section.append(Spacer(1, 0.8 * cm))
        return section
