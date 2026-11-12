"""
Unified Artifact Detection and Restoration — Earth Observation Studio
Entry point:  python -m streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from components.styles import inject_global_css, COLORS
from services.model_service import model_status

st.set_page_config(
    page_title="Unified Artifact Detection and Restoration",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

NAV = [
    ("overview",          "Overview"),
    ("restoration",       "Image Restoration"),
    ("artifact_analysis", "Artifact Analysis"),
    ("performance",       "Performance Reports"),
    ("history",           "Restoration History"),
    ("about",             "About the Model"),
]

NAV_ICONS = {
    "overview":          "◉",
    "restoration":       "◈",
    "artifact_analysis": "◆",
    "performance":       "◇",
    "history":           "≡",
    "about":             "ℹ",
}

with st.sidebar:
    # ── Brand ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding:1rem 0.6rem 0.6rem 0.6rem;">
            <div style="display:flex;align-items:center;gap:0.45rem;">
                <span style="font-size:1.4rem;line-height:1;">🛰️</span>
                <div>
                    <div style="font-size:1rem;font-weight:700;color:#205B59;
                                letter-spacing:-0.01em;line-height:1.1;">Unified Artifact Detection and Restoration</div>
                    <div style="display:flex;align-items:center;gap:0.35rem;margin-top:0.1rem;">
                        <span style="font-size:0.62rem;font-weight:700;
                                     color:#287C78;background:#D5E5E2;
                                     border:1px solid #A0C4C0;border-radius:3px;
                                     padding:0.05rem 0.3rem;letter-spacing:0.05em;">V5</span>
                        <span style="font-size:0.62rem;color:#62747D;
                                     letter-spacing:0.03em;">Earth Observation Studio</span>
                    </div>
                </div>
            </div>
        </div>
        <div style="height:1px;background:#C8D6DA;margin:0 0.4rem 0.4rem 0.4rem;"></div>
        """,
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "overview"

    current = st.session_state.page

    for key, label in NAV:
        icon    = NAV_ICONS.get(key, "·")
        active  = current == key
        if active:
            st.markdown(
                f"""<div style="background:#287C78;border-radius:5px;
                               margin:0.08rem 0.3rem;padding:0.08rem 0;">""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:0.4rem;
                               padding:0.32rem 0.65rem;">
                        <span style="color:#FFFFFF;font-size:0.72rem;">{icon}</span>
                        <span style="color:#FFFFFF;font-size:0.82rem;font-weight:600;">
                            {label}</span>
                    </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            # Invisible button to keep routing logic
            clicked = st.button(label, key=f"nav_{key}", use_container_width=True)
            # Hide the actual button — the HTML above provides the visual
            st.markdown(
                "<style>[data-testid='nav_" + key + "'] {display:none !important;}</style>",
                unsafe_allow_html=True,
            )
        else:
            clicked = st.button(label, key=f"nav_{key}", use_container_width=True)

        if clicked and current != key:
            st.session_state.page = key
            st.rerun()

    # ── Sidebar footer ────────────────────────────────────────────────────────
    st.markdown(
        "<div style='height:1px;background:#C8D6DA;margin:0.6rem 0.4rem 0.5rem 0.4rem;'></div>",
        unsafe_allow_html=True,
    )
    status = model_status()
    if status["loaded"]:
        st.markdown(
            f"<div style='font-size:0.68rem;color:#62747D;padding:0 0.6rem;'>"
            f"<span style='color:#42836B;'>●</span>"
            f" Model ready &nbsp;·&nbsp; <span style='color:#8EA4AE;'>"
            f"{status['device']}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:0.68rem;color:#B75E5E;padding:0 0.6rem;'>"
            "● Model not loaded</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div style='font-size:0.62rem;color:#8EA4AE;padding:0 0.6rem;"
        "margin-top:0.35rem;'>© 2025 ReliefLink</div>",
        unsafe_allow_html=True,
    )

# ── Route ─────────────────────────────────────────────────────────────────────
page = st.session_state.get("page", "overview")

if page == "overview":
    from pages_rl import overview
    overview.render()
elif page == "restoration":
    from pages_rl import restoration
    restoration.render()
elif page == "artifact_analysis":
    from pages_rl import artifact_analysis
    artifact_analysis.render()
elif page == "performance":
    from pages_rl import performance
    performance.render()
elif page == "history":
    from pages_rl import history
    history.render()
elif page == "about":
    from pages_rl import about
    about.render()
