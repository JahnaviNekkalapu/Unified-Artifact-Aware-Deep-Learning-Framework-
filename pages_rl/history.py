"""
Restoration History —Artifact Detection and Quality Metrics

Reads from SQLite. No fabricated records.
All functional logic preserved; visual presentation updated to
Earth Observation Studio theme.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from components.styles import inject_global_css, section_header, badge, CHART
from services.history_service import fetch_all_runs, _get_connection

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

ARTIFACT_HUMAN = {
    "gaussian": "Gaussian Noise",
    "speckle":  "Speckle Noise",
    "blur":     "Blur",
    "haze":     "Haze",
    "cloud":    "Cloud / Occlusion",
    "stripe":   "Stripe Noise",
    "missing":  "Missing / Sensor Gap",
}


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d  %H:%M")
    except Exception:
        return ts


def _fmt_dur(d: float | None) -> str:
    return f"{d:.2f} s" if d is not None else "—"


def _make_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    keys = [
        "run_id", "timestamp", "input_filename", "width", "height",
        "input_format", "duration_s", "status", "artifact_label",
        "artifact_score", "psnr_db", "ssim", "mae", "rmse", "sam",
    ]
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
    return buf.getvalue().encode()


def _activity_chart(runs: list[dict]):
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
            marker=dict(color=CHART["teal"], size=6,
                        line=dict(color="#FFFFFF", width=1.5)),
            fill="tozeroy", fillcolor="rgba(40,124,120,0.10)",
            hovertemplate="%{x}<br>Runs: %{y}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=CHART["bg"], plot_bgcolor=CHART["plot_bg"],
            font=dict(color=CHART["text"], family=CHART["font"], size=11),
            xaxis=dict(gridcolor=CHART["grid"],
                       tickfont=dict(color=CHART["text2"], size=10),
                       showline=False, zeroline=False),
            yaxis=dict(gridcolor=CHART["grid"],
                       tickfont=dict(color=CHART["text2"], size=10),
                       zeroline=True, zerolinecolor=CHART["grid"],
                       title=dict(text="Runs", font=dict(size=10, color=CHART["text2"])),
                       dtick=1),
            margin=dict(l=36, r=16, t=10, b=32),
            height=170, showlegend=False,
        )
        return fig
    except ImportError:
        return None


def render() -> None:
    inject_global_css()

    st.markdown(
        """
        <h1 style="margin-bottom:0.1rem;">Restoration History</h1>
        <p style="color:#62747D;font-size:0.875rem;margin-bottom:1.4rem;">
            All completed restoration runs, stored locally in SQLite.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    section_header("Filter and Sort")
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        search = st.text_input(
            "Search by filename",
            placeholder="Filter by filename…",
            label_visibility="collapsed",
        )
    with fc2:
        sort_by = st.selectbox(
            "Sort", [
                "Newest first", "Oldest first",
                "Duration — slowest first", "Duration — fastest first",
            ],
            label_visibility="collapsed",
        )

    runs = fetch_all_runs(limit=500, search=search)

    if sort_by == "Oldest first":
        runs = list(reversed(runs))
    elif sort_by == "Duration — slowest first":
        runs.sort(key=lambda r: r.get("duration_s") or 0, reverse=True)
    elif sort_by == "Duration — fastest first":
        runs.sort(key=lambda r: r.get("duration_s") or 0)

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # ── Empty state ───────────────────────────────────────────────────────────
    if not runs:
        st.markdown(
            """<div class="rl-card" style="text-align:center;padding:2.5rem;
                color:#8EA4AE;font-size:0.875rem;">
                No restoration runs recorded yet.<br>
                <span style="font-size:0.8rem;">Go to
                <strong style="color:#287C78;">Image Restoration</strong>
                to run your first job.</span>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # ── KPI summary ───────────────────────────────────────────────────────────
    total     = len(runs)
    successes = sum(1 for r in runs if r.get("status") == "success")
    failures  = total - successes

    s1, s2, s3, _ = st.columns(4)
    s1.metric("Matching records", total)
    s2.metric("Successful",       successes)
    s3.metric("Failed",           failures)

    # ── Activity chart ────────────────────────────────────────────────────────
    if len(runs) >= 2:
        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
        section_header("Processing Activity")
        fig = _activity_chart(runs)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    section_header("Run Records")

    hcols = st.columns([0.4, 2.8, 2.2, 0.8, 1.0, 2.8])
    for col, lbl in zip(hcols, ["#", "Filename", "Timestamp", "Status", "Duration", "Top artifact"]):
        col.markdown(
            f"<div style='font-size:0.65rem;font-weight:700;color:#62747D;"
            f"text-transform:uppercase;letter-spacing:0.09em;'>{lbl}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div style='height:1px;background:#D3DEE2;margin:0.3rem 0 0.35rem;'></div>",
        unsafe_allow_html=True,
    )

    for i, run in enumerate(runs, start=1):
        rc = st.columns([0.4, 2.8, 2.2, 0.8, 1.0, 2.8])
        rc[0].markdown(
            f"<span style='font-size:0.75rem;color:#8EA4AE;'>{i}</span>",
            unsafe_allow_html=True,
        )
        rc[1].markdown(
            f"<span style='font-size:0.84rem;color:#253640;'>"
            f"{run.get('input_filename') or '—'}</span>",
            unsafe_allow_html=True,
        )
        rc[2].markdown(
            f"<span style='font-size:0.81rem;color:#62747D;'>"
            f"{_fmt_ts(run['timestamp'])}</span>",
            unsafe_allow_html=True,
        )
        ok = run.get("status") == "success"
        rc[3].markdown(
            badge("OK", "ok") if ok else badge("Err", "err"),
            unsafe_allow_html=True,
        )
        rc[4].markdown(
            f"<span style='font-size:0.81rem;color:#62747D;'>"
            f"{_fmt_dur(run.get('duration_s'))}</span>",
            unsafe_allow_html=True,
        )
        art    = run.get("artifact_label")
        ascore = run.get("artifact_score")
        if art and ascore is not None:
            rc[5].markdown(
                f"<span style='font-size:0.81rem;color:#205B59;'>"
                f"{ARTIFACT_HUMAN.get(art, art)}"
                f" <span style='color:#62747D;'>({ascore*100:.0f}%)</span></span>",
                unsafe_allow_html=True,
            )
        else:
            rc[5].markdown(
                "<span style='font-size:0.81rem;color:#8EA4AE;'>—</span>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='height:1px;background:#F0F4F5;margin:0.18rem 0;'></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

    # ── Run detail ────────────────────────────────────────────────────────────
    section_header("Run Detail")
    run_labels = [
        f"{r.get('input_filename','?')}  —  {_fmt_ts(r['timestamp'])}  [{r['run_id'][:8]}]"
        for r in runs
    ]
    selected_label = st.selectbox(
        "Select run", options=run_labels, index=0, label_visibility="collapsed",
    )

    if selected_label:
        idx = run_labels.index(selected_label)
        run = runs[idx]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Run ID",   run["run_id"][:12] + "…")
        d2.metric("Filename", run.get("input_filename") or "—")
        d3.metric("Status",   (run.get("status") or "—").title())
        d4.metric("Duration", _fmt_dur(run.get("duration_s")))

        e1, e2, e3 = st.columns(3)
        e1.metric("Dimensions",
                  f"{run.get('width','?')} × {run.get('height','?')} px")
        e2.metric("Format",    run.get("input_format") or "—")
        e3.metric("Timestamp", _fmt_ts(run["timestamp"]))

        art_json = run.get("artifact_json")
        if art_json:
            try:
                art_scores = json.loads(art_json)
                st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
                section_header("Artifact Scores")
                sorted_art = sorted(art_scores.items(), key=lambda x: x[1], reverse=True)
                for key, score in sorted_art:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:0.84rem;padding:0.2rem 0;"
                        f"border-bottom:1px solid #F0F4F5;'>"
                        f"<span style='color:#253640;'>"
                        f"{ARTIFACT_HUMAN.get(key, key)}</span>"
                        f"<span style='color:#287C78;font-weight:600;'>"
                        f"{score*100:.1f}%</span></div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        has_metrics = any(
            run.get(k) is not None
            for k in ("psnr_db", "ssim", "mae", "rmse", "sam")
        )
        if has_metrics:
            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
            section_header("Quality Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            for col, (k, lbl) in zip(
                [m1, m2, m3, m4, m5],
                [("psnr_db","PSNR (dB)"),("ssim","SSIM"),
                 ("mae","MAE"),("rmse","RMSE"),("sam","SAM")],
            ):
                val = run.get(k)
                col.metric(lbl, f"{val:.4f}" if val is not None else "—")

        out_fname = run.get("output_filename")
        if out_fname:
            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
            out_path = OUTPUTS_DIR / out_fname
            if out_path.is_file():
                with open(out_path, "rb") as fh:
                    st.download_button(
                        f"⬇  Download  {out_fname}",
                        data=fh.read(),
                        file_name=out_fname,
                        mime="image/png",
                    )
            else:
                st.markdown(
                    f"<div style='font-size:0.8rem;color:#8EA4AE;'>"
                    f"Output file not found: <code>{out_fname}</code></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

    # ── Export & clear ────────────────────────────────────────────────────────
    section_header("Export and Manage")
    ec1, ec2 = st.columns([2, 1])
    with ec1:
        st.download_button(
            "⬇  Export all runs (CSV)",
            data=_make_csv(runs),
            file_name="relieflink_v5_history.csv",
            mime="text/csv",
        )
    with ec2:
        if st.button("Clear all history"):
            st.session_state["confirm_clear"] = True

    if st.session_state.get("confirm_clear"):
        st.warning(
            "This will permanently remove all history records. "
            "Image files on disk are not deleted."
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Confirm delete", type="primary"):
                with _get_connection() as conn:
                    conn.execute("DELETE FROM restoration_runs")
                    conn.commit()
                st.session_state.pop("confirm_clear", None)
                st.success("History cleared.")
                st.rerun()
        with cc2:
            if st.button("Cancel"):
                st.session_state.pop("confirm_clear", None)
                st.rerun()
