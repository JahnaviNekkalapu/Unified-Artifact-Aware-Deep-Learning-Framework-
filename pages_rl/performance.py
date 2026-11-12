"""
Performance Reports — ReliefLink V5

Tab 1: Historical V5 evaluation (200-case test set, static recorded values).
Tab 2: Current-image full-reference metrics — computed only when a clean
       reference image is provided by the user.
"""

from __future__ import annotations

import csv
import io

import streamlit as st
from PIL import Image

from components.styles import inject_global_css, section_header, CHART
from utils.image_processing import pil_to_float

HISTORICAL = {
    "PSNR_dB": 34.380165,
    "SSIM":    0.921893,
    "MAE":     0.013035,
    "RMSE":    0.020414,
    "SAM":     0.026048,
}

METRIC_META = {
    "PSNR_dB": {
        "label": "PSNR", "unit": "dB", "better": "higher",
        "color": CHART["teal"],
        "help": (
            "Peak Signal-to-Noise Ratio measures pixel-level reconstruction quality "
            "on a logarithmic scale. Higher values indicate lower noise-to-signal. "
            "Typical good restoration: > 30 dB."
        ),
    },
    "SSIM": {
        "label": "SSIM", "unit": "", "better": "higher",
        "color": CHART["teal2"],
        "help": (
            "Structural Similarity Index combines luminance, contrast, and structure. "
            "Range [0, 1]; values above 0.9 indicate high structural fidelity."
        ),
    },
    "MAE": {
        "label": "MAE", "unit": "", "better": "lower",
        "color": CHART["blue"],
        "help": (
            "Mean Absolute Error. Average absolute pixel-level difference normalised "
            "to [0, 1]. Lower is better."
        ),
    },
    "RMSE": {
        "label": "RMSE", "unit": "", "better": "lower",
        "color": CHART["blue2"],
        "help": (
            "Root Mean Squared Error. Penalises large errors more than MAE. "
            "Range [0, 1]. Lower is better."
        ),
    },
    "SAM": {
        "label": "SAM", "unit": "rad", "better": "lower",
        "color": "#6B8E8B",
        "help": (
            "Spectral Angle Mapper. Mean per-pixel spectral angle (radians) between "
            "reference and restored image. Measures colour accuracy. Lower is better."
        ),
    },
}


def _metric_card(key: str, value: float) -> str:
    m     = METRIC_META[key]
    lbl   = m["label"] + (f" ({m['unit']})" if m["unit"] else "")
    note  = "↑ higher is better" if m["better"] == "higher" else "↓ lower is better"
    return (
        f'<div class="rl-card" style="border-top:2px solid {m["color"]};text-align:center;'
        f'padding:0.85rem 0.6rem;">'
        f'<div style="font-size:0.65rem;font-weight:700;color:#62747D;'
        f'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.2rem;">{lbl}</div>'
        f'<div style="font-size:1.3rem;font-weight:700;color:#205B59;">{value:.4f}</div>'
        f'<div style="font-size:0.65rem;color:#8EA4AE;margin-top:0.15rem;">{note}</div>'
        f"</div>"
    )


def _hist_bar_chart():
    try:
        import plotly.graph_objects as go

        keys   = list(HISTORICAL.keys())
        names  = [METRIC_META[k]["label"] for k in keys]
        vals   = list(HISTORICAL.values())
        colors = [METRIC_META[k]["color"] for k in keys]

        fig = go.Figure(go.Bar(
            x=names, y=vals,
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
            textfont=dict(color=CHART["text2"], size=11),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=CHART["bg"],
            plot_bgcolor=CHART["plot_bg"],
            font=dict(color=CHART["text"], family=CHART["font"], size=11),
            xaxis=dict(tickfont=dict(color=CHART["text"], size=12),
                       gridcolor=CHART["grid"], zeroline=False),
            yaxis=dict(tickfont=dict(color=CHART["text2"], size=10),
                       gridcolor=CHART["grid"]),
            title=dict(
                text="V5 Evaluation Metrics — 200-case test set",
                font=dict(size=12, color=CHART["text2"]), x=0,
            ),
            margin=dict(l=10, r=10, t=36, b=10),
            height=310,
            showlegend=False,
        )
        return fig
    except ImportError:
        return None


def _compare_chart(hist_v, curr_v, labels):
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Historical (200-case)", x=labels, y=hist_v,
            marker_color=CHART["teal"], marker_line_width=0,
            text=[f"{v:.4f}" for v in hist_v], textposition="outside",
            textfont=dict(size=10, color=CHART["text2"]),
        ))
        fig.add_trace(go.Bar(
            name="Current image", x=labels, y=curr_v,
            marker_color=CHART["blue"], marker_line_width=0,
            text=[f"{v:.4f}" for v in curr_v], textposition="outside",
            textfont=dict(size=10, color=CHART["text2"]),
        ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor=CHART["bg"],
            plot_bgcolor=CHART["plot_bg"],
            font=dict(color=CHART["text"], family=CHART["font"]),
            xaxis=dict(tickfont=dict(color=CHART["text"]),
                       gridcolor=CHART["grid"], zeroline=False),
            yaxis=dict(tickfont=dict(color=CHART["text2"]),
                       gridcolor=CHART["grid"]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=310,
            legend=dict(bgcolor=CHART["bg"], bordercolor=CHART["grid"],
                        font=dict(size=11, color=CHART["text"])),
        )
        return fig
    except ImportError:
        return None


def _make_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
    return buf.getvalue().encode()


def render() -> None:
    inject_global_css()

    st.markdown(
        """
        <h1 style="margin-bottom:0.1rem;">Performance Reports</h1>
        <p style="color:#62747D;font-size:0.875rem;margin-bottom:1.4rem;">
            Image quality metrics for the V5 restoration model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Historical Evaluation", "Current Image Metrics"])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown(
            '<div class="rl-note">Results from offline evaluation on a held-out dataset '
            "of 200 degraded / clean image pairs. "
            "Not computed from any uploaded image.</div>",
            unsafe_allow_html=True,
        )

        section_header("Metric Summary — Historical V5 Evaluation · 200 Cases")
        cols = st.columns(5)
        for col, (key, val) in zip(cols, HISTORICAL.items()):
            col.markdown(_metric_card(key, val), unsafe_allow_html=True)

        st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

        section_header("Chart")
        fig = _hist_bar_chart()
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.bar_chart({METRIC_META[k]["label"]: v for k, v in HISTORICAL.items()})

        st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

        section_header("Metric Descriptions")
        for key, meta in METRIC_META.items():
            with st.expander(
                f"{meta['label']}  —  {HISTORICAL[key]:.4f}"
                + (f" {meta['unit']}" if meta["unit"] else "")
            ):
                st.markdown(
                    f"<div style='font-size:0.84rem;color:#62747D;line-height:1.65;'>"
                    f"{meta['help']}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

        section_header("Export")
        csv_rows = [
            {
                "Metric":   METRIC_META[k]["label"],
                "Value":    HISTORICAL[k],
                "Unit":     METRIC_META[k]["unit"],
                "Better":   METRIC_META[k]["better"],
                "Dataset":  "200-case test set",
                "Model":    "ReliefLink V5",
            }
            for k in HISTORICAL
        ]
        st.download_button(
            "Download historical metrics (CSV)",
            data=_make_csv(csv_rows),
            file_name="relieflink_v5_historical_metrics.csv",
            mime="text/csv",
        )

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown(
            '<div class="rl-note-warn">Full-reference metrics require a spatially aligned '
            "clean reference image. Upload one below to compute PSNR, SSIM, MAE, RMSE, "
            "and SAM for the most recent restoration.</div>",
            unsafe_allow_html=True,
        )

        if "last_restored_pil" not in st.session_state:
            st.info("No restoration has been performed yet. "
                    "Go to **Image Restoration** first.")
            return

        restored_pil: Image.Image = st.session_state["last_restored_pil"]
        rW, rH = restored_pil.size
        st.markdown(
            f"<div style='font-size:0.78rem;color:#62747D;margin-bottom:0.75rem;'>"
            f"Restored image: {rW} × {rH} px</div>",
            unsafe_allow_html=True,
        )

        ref_upload = st.file_uploader(
            "Upload clean reference image (must match restored dimensions)",
            type=["png", "jpg", "jpeg"],
            key="ref_uploader",
            label_visibility="collapsed",
        )

        if ref_upload is None:
            st.info("Upload a reference image to compute full-reference quality metrics.")
            return

        try:
            ref_pil = Image.open(ref_upload).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read reference image: {exc}")
            return

        refW, refH = ref_pil.size
        if (refW, refH) != (rW, rH):
            st.error(
                f"Dimension mismatch: reference is {refW} × {refH}, "
                f"restored is {rW} × {rH}. Supply an aligned reference."
            )
            return

        if st.button("Compute quality metrics", type="primary"):
            try:
                from utils.metrics import compute_all_metrics

                ref_arr      = pil_to_float(ref_pil)
                restored_arr = pil_to_float(restored_pil)

                with st.spinner("Computing metrics…"):
                    m = compute_all_metrics(ref_arr, restored_arr)

                section_header("Computed Metrics")
                cols = st.columns(5)
                for col, key in zip(cols, HISTORICAL):
                    col.markdown(_metric_card(key, m[key]), unsafe_allow_html=True)

                st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

                section_header("Comparison with Historical Evaluation")
                keys   = list(HISTORICAL.keys())
                labels = [METRIC_META[k]["label"] for k in keys]
                hist_v = [HISTORICAL[k] for k in keys]
                curr_v = [m.get(k, 0) for k in keys]
                fig = _compare_chart(hist_v, curr_v, labels)
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

                st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
                section_header("Export")
                csv_rows = [
                    {
                        "Metric":     METRIC_META[k]["label"],
                        "Current":    m.get(k, ""),
                        "Historical": HISTORICAL[k],
                        "Unit":       METRIC_META[k]["unit"],
                        "Model":      "ReliefLink V5",
                    }
                    for k in HISTORICAL
                ]
                st.download_button(
                    "Download current metrics (CSV)",
                    data=_make_csv(csv_rows),
                    file_name="relieflink_v5_current_metrics.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.error(f"Metric computation failed: {exc}")
