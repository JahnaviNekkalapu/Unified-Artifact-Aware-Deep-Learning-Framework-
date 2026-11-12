"""
About the Model — ReliefLink V5

Documentation-style page using verified values from model_config.json
and the actual architecture source.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from components.styles import inject_global_css, section_header

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model_config.json"

ARTIFACT_META = [
    ("Gaussian Noise",    "Random pixel-level intensity variations from sensor thermal noise or ADC quantisation."),
    ("Speckle Noise",     "Multiplicative coherent-interference noise common in SAR and active-sensor imagery."),
    ("Blur",              "Loss of sharpness from defocus, atmospheric turbulence, or platform motion."),
    ("Haze",              "Aerosol forward-scattering reducing contrast and introducing a translucent veil."),
    ("Cloud / Occlusion", "Partial or total occlusion by clouds or cloud shadows; pixels carry no surface reflectance."),
    ("Stripe Noise",      "Detector-element banding from miscalibration, running in the flight or scan direction."),
    ("Missing / Sensor Gap", "Complete data loss in rectangular regions (SLC-off, inter-detector gaps); pixels set to nodata."),
]

LIMITATIONS = [
    "Patch-based inference may produce subtle tiling artefacts on very large images despite Hann-window blending.",
    "The artifact head performs image-level multi-label classification only. Pixel-level segmentation maps are not available.",
    "Full-reference quality metrics require a clean reference image not available at inference time.",
    "GeoTIFF / rasterio support is planned but not yet implemented. Only PNG and JPG are processed.",
    "The model was trained on synthetic degradations. Performance on novel, unseen degradation types may differ.",
    "Very large images (> 4 000 × 4 000 px) are slow without GPU acceleration.",
]

WORKFLOW = [
    ("Upload",        "User selects a PNG, JPG, or JPEG file."),
    ("Preprocessing", "Normalise pixel values to [0, 1]. Sentinel-2 DNs use DN / 10000."),
    ("Patch tiling",  "Images > 256 px split into overlapping 256 × 256 patches (64 px overlap). Reflection padding at edges."),
    ("Inference",     "Each patch processed by ArtifactAwareUNet. Artifact head runs in parallel and conditions the bottleneck embedding."),
    ("Reconstruction","Overlapping patches blended with a Hann weighting window to suppress visible seams."),
    ("Export",        "Restored image saved as PNG and offered for download. Metadata stored in the SQLite history database."),
]


def render() -> None:
    inject_global_css()

    st.markdown(
        """
        <h1 style="margin-bottom:0.1rem;">About the Model</h1>
        <p style="color:#62747D;font-size:0.875rem;margin-bottom:1.4rem;">
            Verified technical specifications for ReliefLink V5.
        </p>
        """,
        unsafe_allow_html=True,
    )

    cfg  = {}
    perf = {}
    if CONFIG_PATH.is_file():
        try:
            cfg  = json.loads(CONFIG_PATH.read_text())
            perf = cfg.get("performance", {})
        except Exception:
            pass

    # ── Identity ──────────────────────────────────────────────────────────────
    section_header("Model Identity")
    ic = st.columns(6)
    specs = [
        ("Name",         "Unified Artifact-Aware Restoration Network"),
        ("Architecture", "Artifact-Aware U-Net"),
        ("Framework",    "PyTorch"),
        ("Parameters",   "8,096,426"),
        ("Input",        "3-ch RGB · 256 × 256"),
        ("Range",        "0 – 1  (float32)"),
    ]
    for col, (lbl, val) in zip(ic, specs):
        col.markdown(
            f"""<div class="rl-card" style="text-align:center;padding:0.75rem 0.5rem;">
                <div style="font-size:0.65rem;font-weight:700;color:#62747D;
                            text-transform:uppercase;letter-spacing:0.09em;
                            margin-bottom:0.25rem;">{lbl}</div>
                <div style="font-size:0.82rem;font-weight:600;color:#205B59;">
                    {val}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Architecture ──────────────────────────────────────────────────────────
    section_header("Architecture")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown(
            """<div class="rl-card" style="border-top:2px solid #287C78;">
                <div style="font-size:0.7rem;font-weight:700;color:#287C78;
                            text-transform:uppercase;letter-spacing:0.08em;
                            margin-bottom:0.5rem;">Encoder / Decoder</div>
                <ul style="font-size:0.83rem;color:#62747D;margin:0;
                            padding-left:1.2rem;line-height:1.85;">
                    <li>4 × DoubleConv blocks: 32 → 64 → 128 → 256 channels</li>
                    <li>MaxPool2d downsampling (4 stages)</li>
                    <li>Bottleneck: DoubleConv 256 → 512</li>
                    <li>4 × ConvTranspose2d upsampling with skip connections</li>
                    <li>Final 1×1 Conv + Sigmoid → restored RGB</li>
                </ul>
            </div>""",
            unsafe_allow_html=True,
        )
    with ac2:
        st.markdown(
            """<div class="rl-card" style="border-top:2px solid #6489AB;">
                <div style="font-size:0.7rem;font-weight:700;color:#6489AB;
                            text-transform:uppercase;letter-spacing:0.08em;
                            margin-bottom:0.5rem;">Artifact Classification Head</div>
                <ul style="font-size:0.83rem;color:#62747D;margin:0;
                            padding-left:1.2rem;line-height:1.85;">
                    <li>AdaptiveAvgPool2d on the bottleneck feature map</li>
                    <li>FC layers: 512 → 128 → 7 raw logits</li>
                    <li>Sigmoid → 7 independent binary probabilities</li>
                    <li>Artifact embedding projected 7 → 512 and added
                        elementwise to bottleneck (conditions the decoder)</li>
                </ul>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Sentinel-2 ────────────────────────────────────────────────────────────
    section_header("Sentinel-2 RGB Band Mapping")
    sb1, sb2, sb3, sb4 = st.columns([1, 1, 1, 3])
    for col, (ch, band, nm) in zip(
        [sb1, sb2, sb3],
        [("Red", "B04", "~665 nm"), ("Green", "B03", "~560 nm"), ("Blue", "B02", "~490 nm")],
    ):
        col.markdown(
            f"""<div class="rl-card" style="text-align:center;padding:0.75rem 0.5rem;">
                <div style="font-size:0.65rem;font-weight:700;color:#62747D;
                            text-transform:uppercase;letter-spacing:0.08em;">{ch}</div>
                <div style="font-size:0.95rem;font-weight:700;color:#205B59;">{band}</div>
                <div style="font-size:0.7rem;color:#8EA4AE;">{nm}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with sb4:
        st.markdown(
            """<div style="font-size:0.83rem;color:#62747D;line-height:1.7;
                           padding-top:0.3rem;">
                Raw Sentinel-2 digital numbers should be converted to surface
                reflectance before inference:<br>
                <code>reflectance = DN / 10000</code><br>
                <strong style="color:#B58B4C;">
                Do not apply this scaling to ordinary PNG or JPG files.</strong>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Artifact categories ───────────────────────────────────────────────────
    section_header("Supported Artifact Categories")
    ac = st.columns(2)
    for i, (label, desc) in enumerate(ARTIFACT_META):
        with ac[i % 2]:
            st.markdown(
                f"""<div style="padding:0.55rem 0;border-bottom:1px solid #EEF2F3;">
                    <div style="font-weight:600;font-size:0.875rem;color:#253640;">
                        {label}</div>
                    <div style="font-size:0.78rem;color:#62747D;margin-top:0.1rem;
                                line-height:1.5;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Historical evaluation ─────────────────────────────────────────────────
    section_header("Historical Evaluation  ·  200-case test set")
    st.markdown(
        "<p style='font-size:0.8rem;color:#62747D;margin:-0.2rem 0 0.75rem;'>"
        "Recorded offline. Not computed from any uploaded image.</p>",
        unsafe_allow_html=True,
    )
    ev = st.columns(5)
    eval_vals = [
        ("PSNR", f"{perf.get('PSNR_dB', 34.380165):.4f} dB"),
        ("SSIM", f"{perf.get('SSIM',    0.921893):.4f}"),
        ("MAE",  f"{perf.get('MAE',     0.013035):.4f}"),
        ("RMSE", f"{perf.get('RMSE',    0.020414):.4f}"),
        ("SAM",  f"{perf.get('SAM',     0.026048):.4f} rad"),
    ]
    for col, (name, val) in zip(ev, eval_vals):
        col.metric(name, val)

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Restoration workflow ──────────────────────────────────────────────────
    section_header("Restoration Workflow")
    for step, desc in WORKFLOW:
        st.markdown(
            f"""<div style="display:flex;gap:0.75rem;padding:0.5rem 0;
                    border-bottom:1px solid #EEF2F3;align-items:flex-start;">
                <div style="font-weight:600;font-size:0.875rem;color:#205B59;
                            min-width:112px;flex-shrink:0;">{step}</div>
                <div style="font-size:0.83rem;color:#62747D;line-height:1.55;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Known limitations ─────────────────────────────────────────────────────
    section_header("Known Limitations")
    for item in LIMITATIONS:
        st.markdown(
            f"""<div style="display:flex;gap:0.6rem;padding:0.4rem 0;
                    border-bottom:1px solid #EEF2F3;">
                <span style="color:#B58B4C;flex-shrink:0;margin-top:0.05rem;">—</span>
                <span style="font-size:0.83rem;color:#62747D;line-height:1.55;">
                    {item}</span>
            </div>""",
            unsafe_allow_html=True,
        )
