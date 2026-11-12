"""
Image Restoration Workspace — Artifact Detection and Quality Metrics

All inference logic unchanged. Visual presentation updated to
Earth Observation Studio theme.
"""

from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from components.styles import inject_global_css, section_header, badge
from services.model_service import get_restorer
from services.history_service import record_run, new_run_id
from utils.image_processing import restore_image_patched

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

ARTIFACT_LABELS = ["gaussian", "speckle", "blur", "haze", "cloud", "stripe", "missing"]
ARTIFACT_HUMAN = {
    "gaussian": "Gaussian Noise",
    "speckle":  "Speckle Noise",
    "blur":     "Blur",
    "haze":     "Haze",
    "cloud":    "Cloud / Occlusion",
    "stripe":   "Stripe Noise",
    "missing":  "Missing / Sensor Gap",
}


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render() -> None:
    inject_global_css()

    st.markdown(
        """
        <h1 style="margin-bottom:0.1rem;">Image Restoration</h1>
        <p style="color:#62747D;font-size:0.875rem;margin-bottom:1.4rem;">
            Restore degraded imagery using the ReliefLink V5 model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        with st.spinner("Initialising model…"):
            restorer = get_restorer()
    except Exception as exc:
        st.error(f"Model failed to load: {exc}")
        st.caption("Verify that `model/best_damage_aware_v5_model.pth` exists.")
        return

    st.markdown(
        f"<div style='font-size:0.73rem;color:#42836B;margin-bottom:1.1rem;'>"
        f"● V5 model ready &nbsp;·&nbsp; {restorer.device}</div>",
        unsafe_allow_html=True,
    )

    # ── Upload panel ──────────────────────────────────────────────────────────
    section_header("Upload Image")
    uploaded = st.file_uploader(
        "Select a PNG, JPG, or JPEG satellite image",
        type=["png", "jpg", "jpeg"],
        help="GeoTIFF / Sentinel-2 support is planned. See About the Model.",
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            """<div style="background:#FFFFFF;border:1.5px dashed #A0BCC0;
                border-radius:8px;padding:2.5rem 1rem;text-align:center;
                color:#8EA4AE;font-size:0.875rem;
                box-shadow:0 1px 3px rgba(37,54,64,0.06);">
                Select a PNG or JPG image above to begin restoration.
            </div>""",
            unsafe_allow_html=True,
        )
        if "last_restored_pil" in st.session_state:
            st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
            section_header("Previous Session Result")
            _show_comparison(
                st.session_state["last_original_pil"],
                st.session_state["last_restored_pil"],
                st.session_state.get("last_elapsed"),
                label="Previous",
            )
            st.download_button(
                "Download previous restored image (PNG)",
                data=_to_png_bytes(st.session_state["last_restored_pil"]),
                file_name="relieflink_v5_previous_restored.png",
                mime="image/png",
            )
        return

    # ── Image info ────────────────────────────────────────────────────────────
    try:
        original_pil = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read image: {exc}")
        return

    W, H     = original_pil.size
    fsize_kb = uploaded.size / 1024

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    section_header("File Information")
    fi1, fi2, fi3, fi4 = st.columns(4)
    fi1.metric("Filename",   uploaded.name)
    fi2.metric("Dimensions", f"{W} × {H} px")
    fi3.metric("File size",  f"{fsize_kb:.1f} KB")
    fi4.metric("Color mode", "RGB")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── Preview ───────────────────────────────────────────────────────────────
    section_header("Input Preview")
    prev_w  = min(W, 520)
    prev_h  = int(H * prev_w / W)
    preview = original_pil.resize((prev_w, prev_h), Image.Resampling.LANCZOS)

    left_col, _ = st.columns([3, 1])
    with left_col:
        st.image(preview, caption=f"Uploaded image  ·  {W} × {H} px",
                 use_container_width=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── Restore controls ──────────────────────────────────────────────────────
    section_header("Run Restoration")

    if W * H > 2048 * 2048:
        st.markdown(
            f'<div class="rl-note-warn">Large image ({W} × {H} px). '
            "Patch-based inference will be applied — this may take several seconds.</div>",
            unsafe_allow_html=True,
        )

    run_btn = st.button("Restore Image", type="primary")

    if run_btn:
        run_id      = new_run_id()
        status_ph   = st.empty()
        progress_ph = st.progress(0)

        try:
            status_ph.markdown(
                "<div style='font-size:0.82rem;color:#62747D;'>Preparing image…</div>",
                unsafe_allow_html=True,
            )
            progress_ph.progress(10)
            t0 = time.perf_counter()

            status_ph.markdown(
                "<div style='font-size:0.82rem;color:#287C78;'>Running V5 inference…</div>",
                unsafe_allow_html=True,
            )
            progress_ph.progress(40)

            restored_pil, avg_logits = restore_image_patched(restorer, original_pil)

            progress_ph.progress(85)
            status_ph.markdown(
                "<div style='font-size:0.82rem;color:#62747D;'>Finalising…</div>",
                unsafe_allow_html=True,
            )
            elapsed = time.perf_counter() - t0
            progress_ph.progress(100)
            status_ph.empty()
            progress_ph.empty()

        except Exception as exc:
            status_ph.empty()
            progress_ph.empty()
            st.error(f"Inference failed: {exc}")
            record_run(
                run_id=run_id,
                input_filename=uploaded.name,
                width=W, height=H,
                input_format=uploaded.type or "image/png",
                duration_s=0.0, status="failed",
            )
            return

        # Save output
        out_filename = f"restored_{run_id[:8]}_{uploaded.name}"
        restored_pil.save(str(OUTPUTS_DIR / out_filename), format="PNG")

        # Artifact probs via sigmoid
        probs = 1.0 / (1.0 + np.exp(-avg_logits.flatten()))
        artifact_scores = {
            label: float(probs[i]) for i, label in enumerate(ARTIFACT_LABELS)
        }
        top_artifact = max(artifact_scores, key=artifact_scores.__getitem__)

        # Persist
        record_run(
            run_id=run_id,
            input_filename=uploaded.name,
            width=W, height=H,
            input_format=uploaded.type or "image/png",
            duration_s=elapsed, status="success",
            artifact_scores=artifact_scores,
            output_filename=out_filename,
        )

        # Session state for cross-page access
        st.session_state["last_artifact_scores"] = artifact_scores
        st.session_state["last_artifact_logits"] = avg_logits
        st.session_state["last_run_id"]          = run_id
        st.session_state["last_restored_pil"]    = restored_pil
        st.session_state["last_original_pil"]    = original_pil
        st.session_state["last_elapsed"]         = elapsed

        # ── Success banner ────────────────────────────────────────────────────
        st.markdown(
            f"""<div style="background:#D5E5E2;border:1px solid #A0C4C0;
                    border-radius:6px;padding:0.6rem 1rem;margin-bottom:1rem;
                    display:flex;align-items:center;justify-content:space-between;
                    flex-wrap:wrap;gap:0.5rem;">
                <span style="font-size:0.875rem;font-weight:600;color:#205B59;">
                    ✓ Restoration complete</span>
                <span style="font-size:0.76rem;color:#62747D;">
                    {elapsed:.2f} s &nbsp;·&nbsp; {W} × {H} px
                    &nbsp;·&nbsp; run {run_id[:8]}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Before / after ────────────────────────────────────────────────────
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        section_header("Before and After")
        _show_comparison(original_pil, restored_pil, elapsed)

        # ── Download ──────────────────────────────────────────────────────────
        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇  Download restored image (PNG)",
            data=_to_png_bytes(restored_pil),
            file_name=f"relieflink_v5_{uploaded.name.rsplit('.', 1)[0]}_restored.png",
            mime="image/png",
        )

        # ── Top artifact ──────────────────────────────────────────────────────
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        section_header("Top Artifact Detected")
        top_score = artifact_scores[top_artifact]
        pct       = top_score * 100
        st.markdown(
            f"""<div class="rl-card"
                    style="border-left:3px solid #287C78;
                           display:flex;align-items:center;
                           justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
                <div>
                    <div style="font-weight:600;font-size:0.9rem;color:#205B59;">
                        {ARTIFACT_HUMAN.get(top_artifact, top_artifact.title())}
                    </div>
                    <div style="font-size:0.76rem;color:#62747D;margin-top:0.15rem;">
                        Confidence: {pct:.1f}% &nbsp;·&nbsp;
                        Open Artifact Analysis for the full breakdown.
                    </div>
                </div>
                <div style="font-size:1.5rem;font-weight:700;color:#287C78;">
                    {pct:.1f}%
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Metrics note ──────────────────────────────────────────────────────
        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="rl-note">Full-reference quality metrics (PSNR, SSIM, MAE, RMSE, SAM) '
            "require a spatially aligned clean reference image. "
            "Upload one in <strong>Performance Reports</strong> to compute them.</div>",
            unsafe_allow_html=True,
        )

    elif "last_restored_pil" in st.session_state and "last_original_pil" in st.session_state:
        st.markdown(
            "<div style='font-size:0.8rem;color:#62747D;margin-bottom:0.75rem;'>"
            "Previous result shown below. Upload a new image and click "
            "<strong>Restore Image</strong> to run again.</div>",
            unsafe_allow_html=True,
        )
        _show_comparison(
            st.session_state["last_original_pil"],
            st.session_state["last_restored_pil"],
            st.session_state.get("last_elapsed"),
            label="Previous",
        )
        st.download_button(
            "⬇  Download previous restored image (PNG)",
            data=_to_png_bytes(st.session_state["last_restored_pil"]),
            file_name="relieflink_v5_previous_restored.png",
            mime="image/png",
        )


def _show_comparison(
    orig: Image.Image,
    rest: Image.Image,
    elapsed: float | None = None,
    label: str = "",
) -> None:
    col_l, col_r = st.columns(2)
    prefix = f"{label} i" if label else "I"
    with col_l:
        st.markdown(
            f"<div style='font-size:0.7rem;font-weight:700;color:#62747D;"
            f"text-transform:uppercase;letter-spacing:0.08em;"
            f"margin-bottom:0.3rem;'>{prefix}nput — degraded</div>",
            unsafe_allow_html=True,
        )
        W, H = orig.size
        st.image(orig, caption=f"{W} × {H} px", use_container_width=True)
    with col_r:
        st.markdown(
            f"<div style='font-size:0.7rem;font-weight:700;color:#287C78;"
            f"text-transform:uppercase;letter-spacing:0.08em;"
            f"margin-bottom:0.3rem;'>V5 output — restored</div>",
            unsafe_allow_html=True,
        )
        rW, rH = rest.size
        cap = f"{rW} × {rH} px"
        if elapsed is not None:
            cap += f"  ·  {elapsed:.2f} s"
        st.image(rest, caption=cap, use_container_width=True)
