"""
Cached model loader for ReliefLink V5.

The restorer is loaded once per Streamlit process using st.cache_resource
and re-used across all pages and user interactions.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from inference.inference import ReliefLinkRestoration

# Use the checkpoint in model/ (matches the original app.py)
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "model" / "best_damage_aware_v5_model.pth"
)


@st.cache_resource(show_spinner=False)
def get_restorer(model_path: str | None = None) -> ReliefLinkRestoration:
    """
    Load (or return the already-loaded) ReliefLinkRestoration instance.

    Raises
    ------
    FileNotFoundError  – when the checkpoint file is missing.
    RuntimeError       – when torch.load or load_state_dict fail.
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return ReliefLinkRestoration(str(path))


def model_status() -> dict:
    """
    Return a lightweight status dict without crashing the calling page.
    Never raises; errors are captured in the dict.
    """
    try:
        r = get_restorer()
        return {
            "loaded":  True,
            "device":  str(r.device),
            "path":    str(_DEFAULT_MODEL_PATH),
            "error":   None,
        }
    except Exception as exc:
        return {
            "loaded":  False,
            "device":  "N/A",
            "path":    str(_DEFAULT_MODEL_PATH),
            "error":   str(exc),
        }
