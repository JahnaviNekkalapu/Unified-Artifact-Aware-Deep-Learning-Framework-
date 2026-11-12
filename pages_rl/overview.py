"""
Overview — Earth Observation Studio  ·  ReliefLink V5
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import streamlit as st

from components.styles import inject_global_css, section_header, kpi_card, badge, CHART
from services.model_service import model_status
from services.history_service import summary_stats, fetch_all_runs

# ── Constants ─────────────────────────────────────────────────────────────────
EVAL = {
    "PSNR":  (34.380165, "dB",  "↑ higher"),
    "SSIM":  (0.921893,  "",    "↑ higher"),
    "MAE":   (0.013035,  "",    "↓ lower"),
    "RMSE":  (0.020414,  "",    "↓ lower"),
    "SAM":   (0.026048,  "rad", "↓ lower"),
}

ARTIFACT_HUMAN = {
    "gaussian": "Gaussian Noise",
    "speckle":  "Speckle Noise",
    "blur":     "Blur",
    "haze":     "Haze",
    "cloud":    "Cloud / Occlusion",
    "stripe":   "Stripe Noise",
    "missing":  "Missing / Sensor Gap",
}

WORKFLOW = [
    ("📤", "Upload",       "PNG / JPG / JPEG"),
    ("⚙️", "Preprocess",  "Normalise · Tile patches"),
    ("🧠", "V5 Inference", "Artifact-Aware U-Net"),
    ("🔍", "Analysis",    "7-class classification"),
    ("📥", "Export",       "PNG · History · CSV"),
]


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%d %b %Y  %H:%M")
    except Exception:
        return ts


def _fmt_dur(d: float | None) -> str:
    return f"{d:.2f} s" if d is not None else "—"


def _activity_chart(runs: list[dict]):
    """Line/area chart of restoration runs per day."""
    try:
        import plotly.graph_objects as go
        from collections import defaultdict

        day_counts: dict = defaultdict(int)
        for r in runs:
            try:
                day = datetime.fromisoformat(r["timestamp"]).strftime("%Y-%m-%d")
                day_counts[day] += 1
            except Exception:
                pass

        if not day_counts:
            return None

        days   = sorted(day_counts)
        counts = [day_counts[d] for d in days]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days, y=counts,
            mode="lines+markers",
            line=dict(color=CHART["teal"], width=2.5),
            marker=dict(color=CHART["teal"], size=6, line=dict(color="#FFFFFF", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(40,124,120,0.10)",
            hovertemplate="%{x}<br>Runs: %{y}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=CHART["bg"],
            plot_bgcolor=CHART["plot_bg"],
            font=dict(color=CHART["text"], family=CHART["font"], size=11),
            xaxis=dict(
                gridcolor=CHART["grid"], tickfont=dict(color=CHART["text2"], size=10),
                showline=False, zeroline=False,
            ),
            yaxis=dict(
                gridcolor=CHART["grid"], tickfont=dict(color=CHART["text2"], size=10),
                zeroline=True, zerolinecolor=CHART["grid"],
                title=dict(text="Runs", font=dict(size=10, color=CHART["text2"])),
                dtick=1,
            ),
            margin=dict(l=40, r=16, t=12, b=32),
            height=200,
            showlegend=False,
        )
        return fig
    except ImportError:
        return None


def _artifact_dist_chart(runs: list[dict]):
    """Horizontal bar chart of top-predicted artifact categories."""
    try:
        import plotly.graph_objects as go

        counts: dict[str, int] = Counter()
        for r in runs:
            if r.get("artifact_label") and r.get("status") == "success":
                counts[r["artifact_label"]] += 1

        if not counts:
            return None

        all_keys   = list(ARTIFACT_HUMAN.keys())
        labels     = [ARTIFACT_HUMAN.get(k, k) for k in all_keys]
        values     = [counts.get(k, 0) for k in all_keys]

        # Sort by count descending
        paired = sorted(zip(values, labels), reverse=True)
        values = [p[0] for p in paired]
        labels = [p[1] for p in paired]

        # Color: teal for non-zero, light blue for zero
        colors = [CHART["teal"] if v > 0 else CHART["grid"] for v in values]

        fig = go.Figure(go.Bar(
            x=values, y=labels,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[str(v) if v > 0 else "" for v in values],
            textposition="outside",
            textfont=dict(size=10, color=CHART["text2"]),
            hovertemplate="%{y}: %{x} run(s)<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=CHART["bg"],
            plot_bgcolor=CHART["plot_bg"],
            font=dict(color=CHART["text"], family=CHART["font"], size=11),
            xaxis=dict(
                gridcolor=CHART["grid"],
                tickfont=dict(color=CHART["text2"], size=10),
                zeroline=False, dtick=1,
                title=dict(text="Run count", font=dict(size=10, color=CHART["text2"])),
            ),
            yaxis=dict(
                tickfont=dict(color=CHART["text"], size=11),
                autorange="reversed",
            ),
            margin=dict(l=0, r=40, t=10, b=30),
            height=250,
            showlegend=False,
        )
        return fig
    except ImportError:
        return None


def render() -> None:
    inject_global_css()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="margin-bottom:0.1rem;">Earth Observation Overview</h1>
        <p style="color:#62747D;font-size:0.875rem;margin-bottom:1.4rem;">
            Monitor satellite image restoration, inspect artifact predictions,
            and review model evaluation results.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    status = model_status()
    stats  = summary_stats()
    runs   = fetch_all_runs(limit=500)

    # ── A. KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            kpi_card("🔁", "Total Runs", str(stats["total"]),
                     "All restoration jobs"),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            kpi_card("✅", "Successful", str(stats["success"]),
                     f"of {stats['total']} total"),
            unsafe_allow_html=True,
        )
    with k3:
        avg = stats["avg_duration"]
        val = f"{avg:.2f} s" if avg else "—"
        st.markdown(
            kpi_card("⏱", "Avg. Process Time", val,
                     "Successful runs" if avg else "No data yet"),
            unsafe_allow_html=True,
        )
    with k4:
        if status["loaded"]:
            st.markdown(
                kpi_card("🟢", "Model Status", "Ready",
                         f"Device · {status['device']}"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="rl-kpi" style="border-color:#D9A8A8;">'
                f'<span class="rl-kpi-icon">🔴</span>'
                f'<div class="rl-kpi-label">Model Status</div>'
                f'<div class="rl-kpi-value" style="color:#B75E5E;font-size:1rem;">Not Loaded</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    if status["error"]:
        st.error(f"Model error: {status['error']}")

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── B & C. Charts (side by side) ──────────────────────────────────────────
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        section_header("Restoration Activity")
        if not runs:
            st.markdown(
                """<div class="rl-card" style="text-align:center;padding:2rem;
                    color:#8EA4AE;font-size:0.85rem;">
                    No restoration runs recorded yet.<br>
                    <span style="font-size:0.78rem;">
                    Restore an image to see activity here.</span></div>""",
                unsafe_allow_html=True,
            )
        else:
            fig = _activity_chart(runs)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Install plotly for interactive charts.")

    with ch2:
        section_header("Artifact Distribution")
        art_runs = [r for r in runs if r.get("artifact_label") and r.get("status") == "success"]
        if not art_runs:
            st.markdown(
                """<div class="rl-card" style="text-align:center;padding:2rem;
                    color:#8EA4AE;font-size:0.85rem;">
                    No artifact predictions yet.<br>
                    <span style="font-size:0.78rem;">
                    Run a restoration to populate this chart.</span></div>""",
                unsafe_allow_html=True,
            )
        else:
            fig = _artifact_dist_chart(runs)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Install plotly for interactive charts.")

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── D. Recent activity table ──────────────────────────────────────────────
    section_header("Recent Activity")
    recents = stats.get("recent", [])

    if not recents:
        st.markdown(
            """<div class="rl-card" style="text-align:center;padding:1.75rem;
                color:#8EA4AE;font-size:0.875rem;">
                No restoration runs recorded yet.
                Upload an image on <strong style="color:#287C78;">Image Restoration</strong>
                to begin.</div>""",
            unsafe_allow_html=True,
        )
    else:
        hcols = st.columns([2.8, 2.4, 0.9, 1.1, 2.8])
        for col, lbl in zip(hcols, ["Filename", "Date / time", "Status", "Duration", "Top artifact"]):
            col.markdown(
                f"<div style='font-size:0.65rem;font-weight:700;color:#62747D;"
                f"text-transform:uppercase;letter-spacing:0.09em;'>{lbl}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='height:1px;background:#D3DEE2;margin:0.3rem 0 0.35rem;'></div>",
            unsafe_allow_html=True,
        )
        for run in recents:
            rc = st.columns([2.8, 2.4, 0.9, 1.1, 2.8])
            rc[0].markdown(
                f"<span style='font-size:0.84rem;color:#253640;'>"
                f"{run.get('input_filename') or '—'}</span>",
                unsafe_allow_html=True,
            )
            rc[1].markdown(
                f"<span style='font-size:0.81rem;color:#62747D;'>"
                f"{_fmt_ts(run['timestamp'])}</span>",
                unsafe_allow_html=True,
            )
            ok = run.get("status") == "success"
            rc[2].markdown(
                badge("OK", "ok") if ok else badge("Err", "err"),
                unsafe_allow_html=True,
            )
            rc[3].markdown(
                f"<span style='font-size:0.81rem;color:#62747D;'>"
                f"{_fmt_dur(run.get('duration_s'))}</span>",
                unsafe_allow_html=True,
            )
            art    = run.get("artifact_label")
            ascore = run.get("artifact_score")
            if art and ascore is not None:
                rc[4].markdown(
                    f"<span style='font-size:0.81rem;color:#205B59;'>"
                    f"{ARTIFACT_HUMAN.get(art, art)} "
                    f"<span style='color:#62747D;'>({ascore*100:.0f}%)</span></span>",
                    unsafe_allow_html=True,
                )
            else:
                rc[4].markdown(
                    "<span style='font-size:0.81rem;color:#8EA4AE;'>—</span>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<div style='height:1px;background:#F0F4F5;margin:0.2rem 0;'></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── E. Historical evaluation ──────────────────────────────────────────────
    section_header("Historical V5 Evaluation  ·  200-case test set")
    st.markdown(
        "<p style='font-size:0.8rem;color:#62747D;margin:-0.25rem 0 0.75rem;'>"
        "Recorded offline on a held-out dataset of 200 degraded / clean pairs. "
        "Not computed from any uploaded image.</p>",
        unsafe_allow_html=True,
    )
    ec = st.columns(5)
    for col, (name, (val, unit, note)) in zip(ec, EVAL.items()):
        display = f"{val:.4f}{(' ' + unit) if unit else ''}"
        col.markdown(
            f"""<div class="rl-card" style="text-align:center;padding:0.8rem 0.5rem;">
                <div style="font-size:0.65rem;font-weight:700;color:#62747D;
                            text-transform:uppercase;letter-spacing:0.09em;
                            margin-bottom:0.2rem;">{name}</div>
                <div style="font-size:1.15rem;font-weight:700;
                            color:#205B59;">{display}</div>
                <div style="font-size:0.65rem;color:#8EA4AE;margin-top:0.15rem;">
                    {note}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── F. Workflow ───────────────────────────────────────────────────────────
    section_header("Restoration Workflow")
    wf_cols = st.columns(len(WORKFLOW))
    for i, (col, (icon, title, sub)) in enumerate(zip(wf_cols, WORKFLOW)):
        sep = " " if i < len(WORKFLOW) - 1 else ""
        col.markdown(
            f"""<div style="background:#FFFFFF;border:1px solid #D3DEE2;
                            border-radius:8px;text-align:center;
                            padding:0.75rem 0.4rem;
                            box-shadow:0 1px 3px rgba(37,54,64,0.06);">
                <div style="font-size:1.2rem;margin-bottom:0.25rem;">{icon}</div>
                <div style="font-size:0.78rem;font-weight:600;color:#205B59;">
                    {title}</div>
                <div style="font-size:0.67rem;color:#62747D;margin-top:0.1rem;">
                    {sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )
