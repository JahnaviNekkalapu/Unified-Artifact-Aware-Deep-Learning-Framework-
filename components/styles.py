"""
ReliefLink V5 — Earth Observation Studio
Global CSS design system.

Palette  ·  atmospheric gray-blue background, cloud-white cards,
            ocean-teal actions, satellite-blue data, dark-teal headings.
"""

import streamlit as st

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    # Surfaces
    "bg":          "#E8EEF0",    # main page background
    "bg2":         "#F0F4F5",    # secondary surface
    "sidebar":     "#E1E9EB",    # sidebar panel
    "card":        "#FFFFFF",    # content cards
    "card_teal":   "#D5E5E2",    # teal-tinted surface
    "card_blue":   "#DCE7F1",    # blue-tinted surface
    # Teal scale
    "teal":        "#287C78",    # primary ocean teal
    "teal_dark":   "#205B59",    # headings / selected nav
    "teal_lt":     "#D5E5E2",    # light teal surface
    "teal_mid":    "#4A9995",    # mid teal
    # Blue scale
    "blue":        "#6489AB",    # satellite blue
    "blue_lt":     "#DCE7F1",    # light blue surface
    "blue_dark":   "#3D5E80",    # dark satellite blue
    # Text
    "text":        "#253640",    # main text
    "text2":       "#62747D",    # secondary text
    "text3":       "#8EA4AE",    # muted text
    # Borders / structure
    "border":      "#D3DEE2",
    "border_lt":   "#E2EAED",
    # Status
    "success":     "#42836B",
    "warning":     "#B58B4C",
    "error":       "#B75E5E",
}

# ── Chart defaults (exported for page use) ────────────────────────────────────
CHART = {
    "bg":       "#FFFFFF",
    "plot_bg":  "#F8FAFB",
    "grid":     "#E2EAED",
    "teal":     "#287C78",
    "blue":     "#6489AB",
    "teal2":    "#4A9995",
    "blue2":    "#8AAEC8",
    "text":     "#253640",
    "text2":    "#62747D",
    "font":     "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
}

_CSS = """
/* ── Global reset ────────────────────────────────────────────────────────── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background-color: #E8EEF0 !important;
    color: #253640 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif !important;
}

[data-testid="stMain"],
section[data-testid="stMain"] > div,
[data-testid="stMainBlockContainer"] {
    background-color: #E8EEF0 !important;
}

.main .block-container {
    background-color: #E8EEF0 !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1240px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #E1E9EB !important;
    border-right: 1px solid #C8D6DA !important;
    min-width: 216px !important;
    max-width: 232px !important;
}
[data-testid="stSidebar"] * {
    color: #253640 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    color: #253640 !important;
    border: none !important;
    border-radius: 5px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.38rem 0.7rem !important;
    text-align: left !important;
    box-shadow: none !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #D0DDE0 !important;
    color: #205B59 !important;
}

/* ── Typography ──────────────────────────────────────────────────────────── */
h1, h2, h3 {
    color: #205B59 !important;
    letter-spacing: -0.01em !important;
}
h1 {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.15rem !important;
}
h2 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
h3 {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}
p, li {
    color: #253640 !important;
    line-height: 1.65 !important;
}

/* ── White content cards ─────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D3DEE2 !important;
    border-radius: 8px !important;
    padding: 0.85rem 0.95rem !important;
    box-shadow: 0 1px 3px rgba(37,54,64,0.06) !important;
}
[data-testid="metric-container"] label {
    color: #62747D !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #253640 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background-color: #287C78 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: 1px solid #205B59 !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.3rem !important;
    font-size: 0.875rem !important;
    box-shadow: 0 1px 2px rgba(40,124,120,0.25) !important;
    transition: background-color 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #205B59 !important;
}
.stButton > button {
    background-color: #FFFFFF !important;
    color: #253640 !important;
    font-weight: 500 !important;
    border: 1px solid #D3DEE2 !important;
    border-radius: 6px !important;
    padding: 0.42rem 1rem !important;
    font-size: 0.85rem !important;
    box-shadow: none !important;
    transition: background-color 0.12s, border-color 0.12s !important;
}
.stButton > button:hover {
    background-color: #F0F4F5 !important;
    border-color: #287C78 !important;
    color: #205B59 !important;
}

/* ── Download buttons ────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: #FFFFFF !important;
    color: #253640 !important;
    border: 1px solid #D3DEE2 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #D5E5E2 !important;
    border-color: #287C78 !important;
    color: #205B59 !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1.5px dashed #A0BCC0 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #287C78 !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    border: 1px solid #D3DEE2 !important;
    border-radius: 6px !important;
    color: #253640 !important;
    font-size: 0.875rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #287C78 !important;
    box-shadow: 0 0 0 2px rgba(40,124,120,0.15) !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #D3DEE2 !important;
    border-radius: 6px !important;
    color: #253640 !important;
    font-size: 0.875rem !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-size: 0.875rem !important;
}

/* ── Progress bar ────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div {
    background-color: #D3DEE2 !important;
    border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div > div,
div[role="progressbar"] > div {
    background-color: #287C78 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #D3DEE2 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #62747D !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.45rem 0.9rem !important;
    background: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #205B59 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #287C78 !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
details {
    background-color: #FFFFFF !important;
    border: 1px solid #D3DEE2 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 3px rgba(37,54,64,0.05) !important;
}
details summary {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #253640 !important;
    padding: 0.6rem 0.9rem !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #D3DEE2 !important;
    margin: 1.1rem 0 !important;
}

/* ── Dataframes ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D3DEE2 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(37,54,64,0.05) !important;
}
[data-testid="stDataFrame"] th {
    background-color: #F0F4F5 !important;
    color: #62747D !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    border-top-color: #287C78 !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track  { background: #E8EEF0; }
::-webkit-scrollbar-thumb  { background: #B0C4CB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #287C78; }

/* ── Image captions ──────────────────────────────────────────────────────── */
[data-testid="stImage"] p {
    color: #62747D !important;
    font-size: 0.75rem !important;
    text-align: center !important;
}

/* ── Plotly chart containers on colored bg ───────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background-color: transparent !important;
}

/* ──────────────────────────────────────────────────────────────────────────
   ReliefLink component classes
   ────────────────────────────────────────────────────────────────────────── */

/* White content card — sits on the gray-blue page */
.rl-card {
    background: #FFFFFF;
    border: 1px solid #D3DEE2;
    border-radius: 8px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(37,54,64,0.06);
}

/* Teal-tinted surface card */
.rl-card-teal {
    background: #D5E5E2;
    border: 1px solid #A0C4C0;
    border-radius: 8px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.8rem;
}

/* Blue-tinted surface card */
.rl-card-blue {
    background: #DCE7F1;
    border: 1px solid #A8C0D6;
    border-radius: 8px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.8rem;
}

/* KPI metric card */
.rl-kpi {
    background: #FFFFFF;
    border: 1px solid #D3DEE2;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    box-shadow: 0 1px 3px rgba(37,54,64,0.06);
}
.rl-kpi-icon {
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
    display: block;
}
.rl-kpi-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #62747D;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.2rem;
}
.rl-kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #205B59;
    line-height: 1.1;
}
.rl-kpi-sub {
    font-size: 0.7rem;
    color: #8EA4AE;
    margin-top: 0.2rem;
}

/* Section label */
.rl-section {
    font-size: 0.66rem;
    font-weight: 700;
    color: #62747D;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    padding-bottom: 0.42rem;
    border-bottom: 1px solid #D3DEE2;
    margin-bottom: 0.7rem;
    margin-top: 0.1rem;
}

/* Inline status badges */
.rl-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.rl-badge-ok {
    background: #D5E5E2;
    color: #42836B;
    border: 1px solid #A0C4C0;
}
.rl-badge-warn {
    background: #F5ECD8;
    color: #B58B4C;
    border: 1px solid #D9C190;
}
.rl-badge-err {
    background: #F5DEDE;
    color: #B75E5E;
    border: 1px solid #D9A8A8;
}
.rl-badge-info {
    background: #DCE7F1;
    color: #3D5E80;
    border: 1px solid #A8C0D6;
}
.rl-badge-neutral {
    background: #F0F4F5;
    color: #62747D;
    border: 1px solid #D3DEE2;
}

/* Inline code */
code {
    background: #EEF3F5 !important;
    color: #205B59 !important;
    border-radius: 3px !important;
    padding: 0.1rem 0.3rem !important;
    font-size: 0.82em !important;
    border: 1px solid #D3DEE2 !important;
}

/* Left-accent info block */
.rl-note {
    background: #F0F4F5;
    border-left: 3px solid #287C78;
    border-radius: 0 6px 6px 0;
    padding: 0.65rem 0.9rem;
    font-size: 0.82rem;
    color: #62747D;
    margin-bottom: 0.9rem;
}
.rl-note-warn {
    background: #F8F2E8;
    border-left: 3px solid #B58B4C;
    border-radius: 0 6px 6px 0;
    padding: 0.65rem 0.9rem;
    font-size: 0.82rem;
    color: #62747D;
    margin-bottom: 0.9rem;
}
"""


def inject_global_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def badge(text: str, kind: str = "info") -> str:
    """kind: ok | warn | err | info | neutral"""
    return f'<span class="rl-badge rl-badge-{kind}">{text}</span>'


# Keep backward-compat aliases used by existing pages
def _compat_badge(text: str, kind: str) -> str:
    mapping = {"success": "ok", "error": "err", "warning": "warn",
               "info": "info", "neutral": "neutral"}
    return badge(text, mapping.get(kind, "info"))


def section_header(title: str) -> None:
    st.markdown(f'<div class="rl-section">{title}</div>', unsafe_allow_html=True)


def kpi_card(icon: str, label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="rl-kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="rl-kpi">'
        f'<span class="rl-kpi-icon">{icon}</span>'
        f'<div class="rl-kpi-label">{label}</div>'
        f'<div class="rl-kpi-value">{value}</div>'
        f"{sub_html}</div>"
    )
