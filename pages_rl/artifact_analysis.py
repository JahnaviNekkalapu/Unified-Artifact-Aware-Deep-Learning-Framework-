"""
Artifact Analysis — ReliefLink V5

Multi-label degradation classification from the model's artifact head.
Raw logits are converted to probabilities via sigmoid (multi-label BCE training).
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from components.styles import inject_global_css, section_header, badge, CHART

ARTIFACT_META = {
    "gaussian": {
        "label": "Gaussian Noise",
        "desc":  "Random pixel-level noise from sensor thermal drift or ADC quantisation. "
                 "Appears as uniform grain distributed across the full image.",
        "color": "#287C78",
    },
    "speckle": {
        "label": "Speckle Noise",
        "desc":  "Multiplicative coherent-interference noise common in SAR imagery. "
                 "Produces a granular texture correlated with scene brightness.",
        "color": "#4A9995",
    },
    "blur": {
        "label": "Blur",
        "desc":  "Loss of sharpness from defocus, atmospheric turbulence, or platform motion. "
                 "Attenuates high-frequency edges and fine spatial structure.",
        "color": "#6489AB",
    },
    "haze": {
        "label": "Haze",
        "desc":  "Aerosol forward-scattering from dust, humidity, or pollution. "
                 "Reduces contrast and drapes a translucent veil over the scene.",
        "color": "#8AAEC8",
    },
    "cloud": {
        "label": "Cloud / Occlusion",
        "desc":  "Partial or total occlusion by clouds or cloud shadows. "
                 "Pixels beneath thick cloud carry no surface-reflectance information.",
        "color": "#3D5E80",
    },
    "stripe": {
        "label": "Stripe Noise",
        "desc":  "Periodic banding from miscalibrated detector elements. "
                 "Lines run parallel to the flight or scan direction.",
        "color": "#B58B4C",
    },
    "missing": {
        "label": "Missing / Sensor Gap",
        "desc":  "Complete data loss in rectangular regions caused by SLC-off failure "
                 "or inter-detector gaps. Pixels are set to zero or a nodata fill.",
        "color": "#B75E5E",
    },
}


def _bar_svg(prob: float, color: str, width: int = 220) -> str:
    """Return a minimal inline SVG progress bar (no plotly needed for inline use)."""
    fill = int(prob * width)
    return (
        f'<svg width="{width}" height="7" style="display:block;margin-top:4px;">'
        f'<rect width="{width}" height="7" rx="3" fill="#E8EEF0"/>'
        f'<rect width="{fill}" height="7" rx="3" fill="{color}"/>'
        f"</svg>"
    )


def _full_chart(sorted_keys: list[str], scores: dict[str, float]):
    """Horizontal bar chart using Plotly."""
    try:
        import plotly.graph_objects as go

        labels = [ARTIFACT_META[k]["label"]  for k in sorted_keys]
        values = [scores[k] * 100            for k in sorted_keys]
        colors = [ARTIFACT_META[k]["color"]  for k in sorted_keys]

        fig = go.Figure(go.Bar(
            x=values, y=labels,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color=CHART["text2"], size=11),
            hovertemplate="%{y}<br>Confidence: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=CHART["bg"],
            plot_bgcolor=CHART["plot_bg"],
            font=dict(color=CHART["text"], family=CHART["font"], size=11),
            xaxis=dict(
                title=dict(text="Confidence (%)", font=dict(size=10, color=CHART["text2"])),
                range=[0, 110],
                gridcolor=CHART["grid"],
                tickfont=dict(color=CHART["text2"], size=10),
                zeroline=False,
            ),
            yaxis=dict(
                tickfont=dict(color=CHART["text"], size=12),
                autorange="reversed",
            ),
            margin=dict(l=0, r=50, t=10, b=30),
            height=290,
            showlegend=False,
        )
        return fig
    except ImportError:
        return None


def render() -> None:
    inject_global_css()

    st.markdown(
        """
        <h1 style="margin-bottom:0.1rem;">Artifact Analysis</h1>
        <p style="color:#62747D;font-size:0.875rem;margin-bottom:1.4rem;">
            Multi-label degradation classification from the V5 model's artifact head.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Empty state ───────────────────────────────────────────────────────────
    if "last_artifact_scores" not in st.session_state:
        st.markdown(
            """<div class="rl-card" style="text-align:center;padding:2.5rem;
                color:#8EA4AE;font-size:0.875rem;">
                No analysis results yet.<br>
                <span style="font-size:0.8rem;">
                Upload and restore an image on
                <strong style="color:#287C78;">Image Restoration</strong>
                first.</span>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    scores   = st.session_state["last_artifact_scores"]
    run_id   = st.session_state.get("last_run_id", "—")
    elapsed  = st.session_state.get("last_elapsed")

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    top_key     = sorted_keys[0]
    top_meta    = ARTIFACT_META[top_key]
    top_score   = scores[top_key]

    # Run reference
    ref = f"Run {str(run_id)[:8]}"
    if elapsed:
        ref += f"  ·  {elapsed:.2f} s"
    st.markdown(
        f"<div style='font-size:0.72rem;color:#8EA4AE;margin-bottom:1rem;'>{ref}</div>",
        unsafe_allow_html=True,
    )

    # ── Layout: top card + chart side-by-side ─────────────────────────────────
    left, right = st.columns([2, 3])

    with left:
        section_header("Top Detected Degradation")
        pct = top_score * 100
        st.markdown(
            f"""<div class="rl-card"
                    style="border-top:3px solid {top_meta['color']};">
                <div style="font-size:1rem;font-weight:700;color:#205B59;
                            margin-bottom:0.35rem;">
                    {top_meta['label']}
                </div>
                <div style="font-size:0.82rem;color:#62747D;line-height:1.55;
                            margin-bottom:0.75rem;">
                    {top_meta['desc']}
                </div>
                <div style="font-size:1.75rem;font-weight:700;
                            color:{top_meta['color']};line-height:1;">
                    {pct:.1f}%
                </div>
                <div style="font-size:0.68rem;color:#8EA4AE;margin-top:0.15rem;">
                    confidence (sigmoid probability)
                </div>
                {_bar_svg(top_score, top_meta['color'], 220)}
            </div>""",
            unsafe_allow_html=True,
        )

        # Prediction summary
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        section_header("Prediction Summary")
        above_50 = [k for k in sorted_keys if scores[k] >= 0.50]
        if above_50:
            for k in above_50:
                m = ARTIFACT_META[k]
                st.markdown(
                    f"<div style='font-size:0.82rem;padding:0.2rem 0;"
                    f"border-bottom:1px solid #F0F4F5;'>"
                    f"<span style='color:#205B59;font-weight:600;'>{m['label']}</span>"
                    f" <span style='color:#8EA4AE;'>—</span> "
                    f"<span style='color:{m['color']};font-weight:600;'>"
                    f"{scores[k]*100:.1f}%</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='font-size:0.82rem;color:#8EA4AE;'>"
                "No class exceeds 50% confidence.</div>",
                unsafe_allow_html=True,
            )

    with right:
        section_header("All Predictions — ranked")
        fig = _full_chart(sorted_keys, scores)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            # Fallback inline bars
            for k in sorted_keys:
                m   = ARTIFACT_META[k]
                s   = scores[k]
                pct = s * 100
                vc  = m["color"] if pct >= 50 else "#8EA4AE"
                st.markdown(
                    f"<div style='padding:0.5rem 0;border-bottom:1px solid #F0F4F5;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;margin-bottom:3px;'>"
                    f"<span style='font-size:0.84rem;color:#253640;'>{m['label']}</span>"
                    f"<span style='font-size:0.84rem;font-weight:600;color:{vc};'>"
                    f"{pct:.1f}%</span></div>"
                    f"{_bar_svg(s, m['color'], 300)}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Full ranked list (compact) ────────────────────────────────────────────
    section_header("All Artifact Categories")
    for k in sorted_keys:
        m   = ARTIFACT_META[k]
        s   = scores[k]
        pct = s * 100
        vc  = m["color"] if pct >= 50 else "#8EA4AE"
        st.markdown(
            f"""<div style="background:#FFFFFF;border:1px solid #D3DEE2;
                    border-radius:6px;padding:0.65rem 0.9rem;margin-bottom:0.5rem;
                    box-shadow:0 1px 2px rgba(37,54,64,0.05);">
                <div style="display:flex;align-items:center;
                            justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:0.875rem;font-weight:600;color:#253640;">
                        {m['label']}</span>
                    <span style="font-size:0.9rem;font-weight:700;color:{vc};">
                        {pct:.1f}%</span>
                </div>
                {_bar_svg(s, m['color'], 600)}
                <div style="font-size:0.74rem;color:#62747D;margin-top:0.4rem;
                            line-height:1.5;">{m['desc']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── Interpretation ────────────────────────────────────────────────────────
    section_header("Interpretation Notes")
    st.markdown(
        """<div class="rl-note">
            <ul style="margin:0;padding-left:1.2rem;line-height:1.85;font-size:0.83rem;">
                <li>Probabilities are produced by sigmoid activation applied to the raw logits
                    from the model's multi-label classification head.</li>
                <li>The head was trained with independent binary cross-entropy per class,
                    so multiple categories may exceed 50% simultaneously.</li>
                <li>Scores ≥ 50% indicate the model detected that degradation type with
                    meaningful confidence.</li>
                <li>The U-Net decoder is conditioned on these predictions via an artifact
                    embedding added to the bottleneck feature map.</li>
                <li>Only image-level classification is available. Pixel-level segmentation
                    maps are not produced by this architecture.</li>
            </ul>
        </div>""",
        unsafe_allow_html=True,
    )
